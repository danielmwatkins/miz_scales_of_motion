"""Used in earlier version of paper - move and consider for use later. Potential to keep if I use it in understanding the deformation!
"""
import cdsapi
import xarray as xr
import os
import numpy as np
import pandas as pd
import pyproj
from urllib.request import urlopen

# Settings
years = np.arange(2003, 2021)
start_date = lambda y: '{y}-03-30 00:00'.format(y=y)
end_date = lambda y: '{y}-10-01 00:00'.format(y=y)
saveloc = '../../data/era5/'

# variables: msl, u10, v10
for year in years:
    c = cdsapi.Client(verify=True)
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
            print('Reading', year)
            savename = str(year) + '_msl_u10_v10'
            ds_rolling = ds.rolling({'time': 24}, center=True).mean(dim='time')
            ds_daily = ds_rolling.sel(time=ds_rolling.time.dt.hour==0)            
            ds_daily.to_netcdf(saveloc + 'era5_' + savename + '.nc',
                         encoding={var: {'zlib': True}
                                      for var in ['msl', 'u10', 'v10',
                                                  'longitude', 'latitude']})
