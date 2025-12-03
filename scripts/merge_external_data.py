"""
* 5-day centered time mean for NSIDC data
* 1-day centered time mean wind velocity
* Estimated distance to ice edge
"""

import os
import numpy as np
import pandas as pd
import pyproj
import sys
import warnings
import xarray as xr

sys.path.append('../scripts/')
from scipy.interpolate import interp2d
from drifter import compute_velocity
from drifter import compute_along_across_components
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

#### Specify locations for data.
# External datasets not included in archive are stored in a lower 
ift_interp_loc = '../data/floe_tracker/ift_floe_trajectories/'
ift_prop_loc = '../data/floe_tracker/ift_floe_property_tables/clean/'
nsidc_motion_loc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_icemotion/'
nsidc_sic_loc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr/aggregate/'
saveloc = '../data/floe_tracker/'

#### Load ift trajectories
ift_tracks = {}
for year in range(2003, 2021):
    df = pd.read_csv(ift_interp_loc + 'ift_interp_floe_trajectories_{y}.csv'.format(y=year),
                    index_col=0)
    df['datetime'] = pd.to_datetime(df['datetime'])
    ift_tracks[year] = df.copy()

#### Load ift property tables
ift_props = {}
for year in range(2003, 2021):
    df = pd.read_csv(ift_prop_loc + 'ift_clean_floe_properties_{y}.csv'.format(y=year),
                    index_col=0)
    df['datetime'] = pd.to_datetime(df['datetime'])
    ift_props[year] = df.copy()
    
#### Functions for interpolating NSIDC data
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
    
def min_distance(x0, y0, comp_df):
    """Compute the distance to the nearest pixel with 0 sea ice concentration"""
    d = (comp_df['x'] - x0)**2 + (comp_df['y'] - y0)**2
    return np.round(np.sqrt(d.min()), 0)

for year in ift_tracks:
    with xr.open_dataset(nsidc_sic_loc + '/seaice_conc_daily_nh_' + \
                         str(year) + '_v04r00.nc') as sic_data:
        ds = xr.Dataset({'sea_ice_concentration':
                         (('time', 'y', 'x'), sic_data['cdr_seaice_conc'].data)},
                   coords={'time': (('time', ), sic_data['time'].data),
                           'x': (('x', ), sic_data['xgrid'].data), 
                           'y': (('y', ), sic_data['ygrid'].data)})

        # pre-allocate
        ift_tracks[year]['edge_dist_km'] = np.nan
        ift_tracks[year]['coast_dist_km'] = np.nan
        ift_props[year]['edge_dist_km'] = np.nan
        ift_props[year]['coast_dist_km'] = np.nan

        # Restrict to greenland sea region for faster search
        xmin=10e3
        xmax=30e5
        ymax=20e3
        ymin=-5.4e6

        # Get coast dist and edge dist for each floe in the interpolated data and the clean initial data
        for date, group in ift_tracks[year].groupby('datetime'):
            sic_subset = ds.sel(time=date.strftime('%Y-%m-%d'), x=slice(xmin, xmax), y=slice(ymax, ymin))

            # Use pixels with 0 SIC to find the distance to the ice edge
            ocean_subset = sic_subset.where(sic_subset['sea_ice_concentration'] == 0)
            comp_df = ocean_subset.to_dataframe().dropna().reset_index()
            if len(comp_df) > 0:
                d = pd.Series([min_distance(x, y, comp_df) for x, y in zip(group.x_stere, group.y_stere)],
                      index=group.index)
                ift_tracks[year].loc[d.index, 'edge_dist_km'] = np.round(d/1e3, 1)

            # Use the land flag in the CDR SIC to get distance to land
            land_subset = sic_subset.where(sic_subset['sea_ice_concentration'] == 2.54)
            comp_df = land_subset.to_dataframe().dropna().reset_index()
            if len(comp_df) > 0:
                d = pd.Series([min_distance(x, y, comp_df) for x, y in zip(group.x_stere, group.y_stere)],
                      index=group.index)
                ift_tracks[year].loc[d.index, 'coast_dist_km'] = np.round(d/1e3, 1)           

        for date, group in ift_props[year].groupby('datetime'):
            sic_subset = ds.sel(time=date.strftime('%Y-%m-%d'), x=slice(xmin, xmax), y=slice(ymax, ymin))

            # Use pixels with 0 SIC to find the distance to the ice edge
            ocean_subset = sic_subset.where(sic_subset['sea_ice_concentration'] == 0)
            comp_df = ocean_subset.to_dataframe().dropna().reset_index()
            if len(comp_df) > 0:
                d = pd.Series([min_distance(x, y, comp_df)
                               for x, y in zip(group.x_stere, group.y_stere)],
                      index=group.index)
                ift_props[year].loc[d.index, 'edge_dist_km'] = np.round(d/1e3, 1)

            # Use the land flag in the CDR SIC to get distance to land
            land_subset = sic_subset.where(sic_subset['sea_ice_concentration'] == 2.54)
            comp_df = land_subset.to_dataframe().dropna().reset_index()
            if len(comp_df) > 0:
                d = pd.Series([min_distance(x, y, comp_df) for x, y in zip(group.x_stere, group.y_stere)],
                      index=group.index)

                ift_props[year].loc[d.index, 'coast_dist_km'] = np.round(d/1e3, 1) 


    ift_props[year].to_csv(saveloc + 'ift_floe_property_tables/with_nsidc/ift_floe_properties_{y}.csv'.format(y=year))     

      
for year in ift_tracks:
    with xr.open_dataset(nsidc_motion_loc + 'icemotion_daily_nh_25km_' + \
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
            icemotion = icemotion_along_track(ift_tracks[year], ds_new, dt_days=time)
            icemotion.index = ift_tracks[year].index
            if 'u_nsidc' not in ift_tracks[year].columns:
                ift_tracks[year] = pd.concat([ift_tracks[year], icemotion], axis=1)
            else:
                ift_tracks[year] = pd.concat([ift_tracks[year],
                                              icemotion.drop(['u_nsidc', 'v_nsidc'], axis=1)], axis=1)
pd.concat([ift_tracks[y] for y in ift_tracks]).to_csv(saveloc + 'ift_floe_trajectories.csv')