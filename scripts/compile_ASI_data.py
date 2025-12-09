import os
import xarray as xr
import pandas as pd
import numpy as np
import rasterio as rio

asi_dataloc = '/Users/dwatkin2/Documents/research/data/ASI_SIC/'
image_dataloc = "../data/modis_images/landmask.tiff"
ref_image = rio.open(image_dataloc)
left, bottom, right, top = ref_image.bounds

n, m = ref_image.shape
X = np.linspace(left, right, m)
Y = np.linspace(bottom, top, n)

year_folders = [f for f in os.listdir(asi_dataloc) if 'asi' in f]

asi_scenes = []
dates = []
for yf in year_folders:
    year_data = []
    year = yf.split('-')[3]
    new_fname = '../data/floe_tracker/ift_floe_property_tables_asi/ift_raw_floe_properties_asi_{y}.csv'.format(y=year)
    

    start_date = pd.to_datetime('{y}-03-31'.format(y=year))
    end_date = pd.to_datetime('{y}-09-30'.format(y=year))
    files = [f for f in os.listdir(asi_dataloc + '/' + yf) if 'asi' in f]
    files.sort()
    for file in files:
        date = pd.to_datetime(file.split('-')[-2])
        if (date >= start_date) & (date <= end_date):
            with xr.open_dataset(asi_dataloc + yf + '/' + file) as ds_asi:
                asi_sel = ds_asi.sel(x=slice(left, right), y=slice(bottom, top))
                asi_scenes.append(asi_sel['z'])
                dates.append(date)
    print('Done with year', year)

x = asi_scenes[0]['x'].data
y = asi_scenes[0]['y'].data
ds = xr.Dataset({'sic': (('time', 'y', 'x'), asi_scenes)}, coords={'time': (('time',),dates),
                                                                  'x': (('x', ), x), 
                                                                  'y': (('y', ), y)})

ds = ds.sortby('time')
ds.to_netcdf('../data/asi_sic_merged.nc', 
             encoding={'sic': {'zlib': True}})