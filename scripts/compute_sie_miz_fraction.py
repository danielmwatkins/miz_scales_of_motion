"""
Compute the sea ice extent (15-100% SIC) and the MIZ extent (15-85%) using the NSIDC SIC CDR v4.
"""

import os
import xarray as xr
import pandas as pd
import numpy as np


data = "/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr_v4/"


left = 247326.41049684677
right = 1115678.4104968468
bottom = -2089839.4681516434
top = -635759.4681516434

def compute_sic(left_x, right_x, lower_y, upper_y, sic_data):
    """Computes the sea ice extent as a fraction of total area within the region bounded
    by <left_x>, <right_x>, <bottom_y>, and <top_y> using the netcdf file <sic_data>. Assumes
    that sic_data is the NSIDC SIC CDR.
    
    SIF: Sea ice area / ocean area for the image.
    SIE: Total area with sea ice greater than 15% and less than or equal to 100%
    MIZE: Total area with sea ice between 15% and 85%.
    """

    x_idx = (sic_data.xgrid >= left_x) & (sic_data.xgrid <= right_x)
    y_idx = (sic_data.ygrid >= lower_y) & (sic_data.ygrid <= upper_y)
    
    with_ice = ((sic_data.sel(x=x_idx, y=y_idx)['cdr_seaice_conc'] > 0.15) & \
                (sic_data.sel(x=x_idx, y=y_idx)['cdr_seaice_conc'] <= 1))
    coast_mask = (sic_data.sel(x=x_idx, y=y_idx)['cdr_seaice_conc'] > 1).sum() 
    total_area_pixels = np.prod(with_ice.shape)
    sic_extent_pixels = with_ice.sum().data

    miz_ice = (sic_data.sel(x=x_idx, y=y_idx)['cdr_seaice_conc'] > 0.15) & \
                (sic_data.sel(x=x_idx, y=y_idx)['cdr_seaice_conc'] <= 0.85)
    miz_extent_pixels = miz_ice.sum().data
    
    sic_mean = (sic_data.sel(x=x_idx, y=y_idx).where(with_ice))['cdr_seaice_conc'].mean().data
    return {'sea_ice_fraction': np.round(sic_extent_pixels/(total_area_pixels - coast_mask.data), 3),
            'mean_sea_ice_concentration': np.round(sic_mean, 3),
            'sea_ice_extent': sic_extent_pixels * 25 * 25, # km2
            'miz_ice_extent': miz_extent_pixels * 25 * 25} # km2

results = {}
for year in range(2003, 2021):
    files = os.listdir(data + str(year))
    for f in files:
        with xr.open_dataset(data + str(year) + '/' + f) as ds:
            results[pd.to_datetime(f.split('_')[4])] = compute_sic(left, right, bottom, top, ds)
    print(year)

results = pd.DataFrame(results).T.dropna()
results = results.loc[results.sea_ice_fraction > 0]
results.sort_index(inplace=True)
results.to_csv('../data/nsidc_greenland_sea_ice_extent.csv')