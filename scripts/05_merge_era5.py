"""Adds the ERA5 data to the IFT table. Optionally, downloads ERA5 data and calculates daily averages"""
import xarray as xr
import os
import numpy as np
import pandas as pd

download = False
# Assume that the era5 daily averages are available already
era5_dataloc = '../../data/era5/'

if download:
    # For each year, select the data from March to end of September,
    # then resample the data to daily averages using a 12 pm to 12 pm window.
    # This section doesn't work yet, though
    import cdsapi
    from urllib.request import urlopen
    
    # Settings
    years = np.arange(2003, 2021)
    start_date = lambda y: '{y}-03-30 00:00'.format(y=y)
    end_date = lambda y: '{y}-10-01 00:00'.format(y=y)
    saveloc = '../../data/era5/'
    
    # variables: msl, u10, v10
    for year in years:
        c = cdsapi.Client(verify=True)
        print('Downloading MSL')
        params = {'product_type': 'reanalysis',
                  'format': 'netcdf',
                  'variable': ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind'],
                  'date': list(pd.date_range(start_date(year), end_date(year), freq='1D').strftime('%Y-%m-%d')),
                  'time': list(map("{:02d}:00".format, range(0,24))),
                  'area': [90, -180, 50, 180]}
        
        fl = c.retrieve('reanalysis-era5-single-levels', params)
        saveloc = '../data/era5/'
        with urlopen(fl.location) as f:
            with xr.open_dataset(f.read()) as ds:
                savename = str(year) + '_msl_u10_v10'
                ds_rolling = ds.rolling({'time': 24}, center=True).mean(dim='time') # centered average
                ds_daily = ds_rolling.sel(time=ds_rolling.time.dt.hour==0) # subset to 0 UTC
                ds_daily.to_netcdf(saveloc + 'era5_' + savename + '.nc', # save compressed version using zlib
                             encoding={var: {'zlib': True}
                                          for var in ['msl', 'u10', 'v10',
                                                      'longitude', 'latitude']})

# TBD: Can I improve this to use bilinear interpolation? Also should add MSL

def era5_uv_along_track(position_data, uv_data):
    """Uses the xarray advanced interpolation to get along-track era5 winds.
    Uses nearest neighbor for now for speed."""
    
    uv = pd.DataFrame(data=np.nan, index=position_data.index, columns=['u_wind', 'v_wind'])
    
    for date, group in position_data.groupby(position_data.datetime):
        
        x = xr.DataArray(group.longitude, dims="z")
        y = xr.DataArray(group.latitude, dims="z")
        U = uv_data.sel(time=date)['u10'].interp(
            {'longitude': x,
             'latitude': y}, method='nearest').data
        V = uv_data.sel(time=date)['v10'].interp(
            {'longitude': x,
             'latitude': y}, method='nearest').data

        uv.loc[group.index, 'u_wind'] = np.round(U.T, 3)
        uv.loc[group.index, 'v_wind'] = np.round(V.T, 3)

    return uv

ift_loc = '../data/floe_tracker/ift_with_nsidc.csv' # Output from the interpolate ift file
ift_data = pd.read_csv(ift_loc, index_col=0)
# ift_data['datetime_temp'] = pd.to_datetime(ift_data['datetime'].values)
ift_data['datetime'] = pd.to_datetime(ift_data['datetime']) #.dt.date)
floe_tracker_results = {year: group for year, group in ift_data.groupby(ift_data['datetime'].dt.year)}

for year in floe_tracker_results:
    floe_tracker_results[year][['u_wind', 'v_wind']] = np.nan
    for month, data in floe_tracker_results[year].groupby(floe_tracker_results[year].datetime.dt.month):
       # Depending on the file name used for the ERA5 data this section will need to be adjusted.
        with xr.open_dataset(era5_dataloc + str(year) + '/' + \
                             'era5_uvmsl_daily_mean_' + \
                             str(year) + '-' + str(month).zfill(2) + '-01.nc') as ds_era:
            floe_tracker_results[year].loc[
                data.index, ['u_wind', 'v_wind']] = era5_uv_along_track(data, ds_era)    

ft_df = pd.concat(floe_tracker_results)
ft_df.drop('datetime', axis=1, inplace=True)
ft_df.rename({'datetime_temp': 'datetime'}, axis=1, inplace=True)
ft_df['wind_speed'] = (ft_df['u_wind']**2 + ft_df['v_wind']**2)**0.5

ft_df.to_csv('../data/floe_tracker/ift_with_era5.csv')