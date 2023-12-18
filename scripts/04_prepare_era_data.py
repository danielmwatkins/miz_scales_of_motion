"""First downloads ERA5 data to file. Then projects onto a polar stereographic grid centered on the North Pole."""
import cdsapi
import xarray as xr
import os
import numpy as np
import pandas as pd
import pyproj
from urllib.request import urlopen
import xesmf as xe

# Settings
start_dates = ['2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01',
               '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01',
               '2020-06-01', '2020-07-01']
end_dates = ['2019-10-31', '2019-11-30', '2019-12-31', '2020-01-31',
               '2020-02-29', '2020-03-31', '2020-04-20', '2020-05-31',
               '2020-06-30', '2020-07-31']
savenames = ['Oct2019', 'Nov2019', 'Dec2019', 'Jan2020', 'Feb2020',
             'Mar2020', 'Apr2020', 'May2020', 'Jun2020', 'Jul2020']

saveloc = '../../data/era5/'



for start_date, end_date, savename in zip(start_dates, end_dates, savenames):
    c = cdsapi.Client(verify=True)
    print('Downloading MSL')
    params = {'product_type': 'reanalysis',
              'format': 'netcdf',
              'variable': ['mean_sea_level_pressure'],
              'date': list(pd.date_range(start_date, end_date, freq='1D').strftime('%Y-%m-%d')),
              'time': list(map("{:02d}:00".format, range(0,24))),
              'area': [90, -180, 50, 180]}
    
    fl = c.retrieve('reanalysis-era5-single-levels', params)
    saveloc = '../data/era5/'
    with urlopen(fl.location) as f:
        with xr.open_dataset(f.read()) as ds:
            ds.to_netcdf(saveloc + 'era5_msl_' + savename + '.nc',
                         encoding={var: {'zlib': True}
                                      for var in ['msl',
                                                  'longitude', 'latitude']})
