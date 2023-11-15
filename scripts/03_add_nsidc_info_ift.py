"""Takes the ice floe tracker data used in the GRL paper and adds NSIDC sea ice motion and sea ice concentration.
TBD: Describe the saved NSIDC data so that others can run this."""

import os
import numpy as np
import pandas as pd
import pyproj
import sys
import warnings
import xarray as xr

sys.path.append('../../../packages/buoy_processing/')
from scipy.interpolate import interp2d
from drifter.src.analysis import compute_velocity
from drifter.src.analysis import compute_along_across_components
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

#### Specify locations for data.
# External datasets not included in archive are stored in a lower 
ift_loc = '../data/floe_tracker/ift_with_zeta.csv' # Output from the interpolate ift file
sic_loc = '../../../data/nsidc_daily_cdr/'
motion_loc = '../../../data/nsidc_daily_icemotion/'
save_loc = '../data/floe_tracker/'

#### Load ift data
df = pd.read_csv(ift_loc)
df['date'] = pd.to_datetime(df.date.values)
df.rename({'Unnamed: 0': 'year', 'date': 'datetime'}, axis=1, inplace=True)
df.drop('Unnamed: 1', axis=1, inplace=True)

#### Functions for interpolating NSIDC data
def sic_along_track(position_data, sic_data):
    """Uses the xarray advanced interpolation to get along-track sic
    via nearest neighbors. Nearest neighbors is preferred because numerical
    flags are used for coasts and open ocean, so interpolation is less meaningful."""
    # Sea ice concentration uses NSIDC NP Stereographic
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
    transformer_stere = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    
    sic = pd.Series(data=np.nan, index=position_data.index)
    
    for date, group in position_data.groupby(position_data.datetime.dt.date):
        x_stere, y_stere = transformer_stere.transform(
            group.longitude, group.latitude)
        
        x = xr.DataArray(x_stere, dims="z")
        y = xr.DataArray(y_stere, dims="z")
        SIC = sic_data.sel(time=date.strftime('%Y-%m-%d'))['sea_ice_concentration'].interp(
            {'x': x,
             'y': y}, method='nearest').data

        sic.loc[group.index] = np.round(SIC.T, 3)
    # sic[sic > 100] = np.nan
    return sic

def min_distance(x0, y0, comp_df):
    """Compute the distance to the nearest pixel with above 0 and less than 30 percent sea ice concentration"""
    d = (comp_df['x'] - x0)**2 + (comp_df['y'] - y0)**2
    return np.round(np.sqrt(d.min()), 0)

def icemotion_along_track(position_data, uv_data, dt_days=5):
    """Uses the xarray advanced interpolation to get along-track ice motion.
    U and V are relative to the polar stereographic grid (currently at least).
    Also calculates 5-day centered mean velocity. 
    """
    
    dt = str(int(dt_days/2)) + 'D' # (Interval is date-dt, date+dt so total is 2*dt + 1 days)
    dt_name = str(dt_days) + 'D'
    uv = pd.DataFrame(data=np.nan,
                      index=position_data.index,
                      columns=['u_nsidc', 'v_nsidc',
                               'u' + dt_name + '_nsidc', 'v' + dt_name + '_nsidc'])
    
    # Ice motion vectors use lambert azimuthal equal area
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs')
    transformer_laea = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
                
    for date, group in position_data.groupby(position_data.datetime):
        x_laea, y_laea = transformer_laea.transform(
            group.longitude, group.latitude)        
        x = xr.DataArray(x_laea, dims="z")
        y = xr.DataArray(y_laea, dims="z")
        U = uv_data.sel(time=date.strftime('%Y-%m-%d'))['u'].interp(
            {'x': x,
             'y': y}, method='linear').data/100
        V = uv_data.sel(time=date.strftime('%Y-%m-%d'))['v'].interp(
            {'x': x,
             'y': y}, method='linear').data/100

        uv.loc[group.index, 'u_nsidc'] = np.round(U.T, 3).squeeze()
        uv.loc[group.index, 'v_nsidc'] = np.round(V.T, 3).squeeze()

        uv_5dmean = uv_data.sel(time=slice(pd.to_datetime(date) - pd.to_timedelta(dt),
                                           pd.to_datetime(date) + pd.to_timedelta(dt))).mean(dim='time')
        
        U = uv_5dmean['u'].interp(
            {'x': x,
             'y': y}, method='linear').data/100
        V = uv_5dmean['v'].interp(
            {'x': x,
             'y': y}, method='linear').data/100

        uv.loc[group.index, 'u' + dt_name + '_nsidc'] = np.round(U.T, 3).squeeze()
        uv.loc[group.index, 'v' + dt_name + '_nsidc'] = np.round(V.T, 3).squeeze()

    return uv


ift_data = {year: group for year, group in df.groupby('year')}
for year in ift_data:
    with xr.open_dataset(sic_loc + '/aggregate/seaice_conc_daily_nh_' + \
                         str(year) + '_v04r00.nc') as sic_data:
        ds = xr.Dataset({'sea_ice_concentration':
                         (('time', 'y', 'x'), sic_data['cdr_seaice_conc'].data)},
                   coords={'time': (('time', ), sic_data['time'].data),
                           'x': (('x', ), sic_data['xgrid'].data), 
                           'y': (('y', ), sic_data['ygrid'].data)})
        sic = sic_along_track(ift_data[year], ds)
        ift_data[year]['sea_ice_concentration'] = sic.astype(float)
        
        ift_data[year]['edge_dist'] = np.nan
        for date, group in ift_data[year].groupby('datetime'):
            # Simple estimate of distance to ice edge
            # based on minimum distance to pixel with between 0 and 30 % sea ice concentration.
            # These values are based on the Greenland Sea region - change it for other locations
            xmin=10e3
            xmax=30e5
            ymax=20e3
            ymin=-5.4e6
            sic_subset = ds.sel(time=date.strftime('%Y-%m-%d'), x=slice(xmin, xmax), y=slice(ymax, ymin))
            sic_subset = sic_subset.where((sic_subset['sea_ice_concentration'] > 0) & (sic_subset['sea_ice_concentration'] < 0.3))
            comp_df = sic_subset.to_dataframe().dropna().reset_index()
            if len(comp_df) > 0:
                d = pd.Series([min_distance(x, y, comp_df) for x, y in zip(group.x_stere, group.y_stere)],
                      index=group.index)
                ift_data[year].loc[d.index, 'edge_dist'] = d
        

    with xr.open_dataset(motion_loc + 'icemotion_daily_nh_25km_' + \
                         str(year) + '0101_' + str(year) + '1231_v4.1.nc') as ds:
        L = np.deg2rad(ds['longitude'])
        u = ds['u'] * np.cos(L)  +  ds['v'] * np.sin(L)
        v = -ds['u'] * np.sin(L)  +  ds['v'] * np.cos(L)
        ds_new = xr.Dataset({'u': (('time', 'y', 'x'), u.data),
                             'v': (('time', 'y', 'x'), v.data)},
                           coords={'time': (('time', ),
                                            pd.to_datetime({'year': ds['time'].dt.year,
                                                            'month': ds['time'].dt.month,
                                                            'day': ds['time'].dt.day})),
                                   'x': (('x', ), ds['x'].data), 
                                   'y': (('y', ), ds['y'].data)})

        for time in [5, 15, 31]:
            icemotion = icemotion_along_track(ift_data[year], ds_new, dt_days=time)
            icemotion.index = ift_data[year].index
            if 'u_nsidc' not in ift_data[year].columns:
                ift_data[year] = pd.concat([ift_data[year], icemotion], axis=1)
            else:
                # check this - it's not quite right. Like, we're getting duplicates of the u/v columns
                ift_data[year] = pd.concat([ift_data[year], icemotion.drop(['u_nsidc', 'v_nsidc'], axis=1]), axis=1)

    ift_data[year] = compute_along_across_components(ift_data[year],
                                     umean='u5D_nsidc',
                                     vmean='v5D_nsidc')

    



#### Save the results
# TBD: Add a readme file explaining the columns and units
# Potentially add circularity and convert the area/perim/axes to km
pd.concat(ift_data).reset_index(drop=True).to_csv('../data/floe_tracker/ift_with_nsidc.csv')