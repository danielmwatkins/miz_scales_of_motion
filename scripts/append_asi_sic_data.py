import ultraplot as uplt
import xarray as xr
import pandas as pd
import os
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_image
import skimage
from skimage.measure import regionprops_table

asi_dataloc = '/Users/dwatkin2/Documents/research/data/ASI_SIC/'
image_dataloc = "/Volumes/Research/ENG_Wilhelmus_Shared/group/IFT_fram_strait_dataset/"

year_folders = [f for f in os.listdir(asi_dataloc) if 'asi' in f]

def get_month_folder(date):
    """Simple utility for navigating file structure"""
    start = '{y}{m}01'.format(y=date.year,
                              m=str(date.month).zfill(2))
    end = '{y}{m}01'.format(y=date.year,
                              m=str(date.month + 1).zfill(2))
    if (date.month == 3) | (date.month == 4):
        start = '{y}0331'.format(y=date.year)
        end = '{y}0501'.format(y=date.year)
    
    return '-'.join(['fram_strait', start, end])

for yf in year_folders:
    year_data = []
    year = yf.split('-')[3]
    print('Starting', year)
    new_fname = '../data/floe_tracker/ift_floe_property_tables_asi/ift_raw_floe_properties_asi_{y}.csv'.format(y=year)
    print(new_fname)
    if not os.path.exists(new_fname):
        start_date = pd.to_datetime('{y}-03-31'.format(y=year))
        end_date = pd.to_datetime('{y}-09-30'.format(y=year))
        files = [f for f in os.listdir(asi_dataloc + '/' + yf) if 'asi' in f]
        files.sort()
        for file in files:
            date = pd.to_datetime(file.split('-')[-2])
            if (date >= start_date) & (date <= end_date):
                if date.day == 1:
                    print(date)
                # get aqua and terra labeled images separately
                try:
                    aqua_img = rio.open(image_dataloc + \
                                        'fram_strait-{y}/{m}/labeled_raw/{d}.{s}.labeled_raw.250m.tiff'.format(
                        d=date.strftime('%Y%m%d'), s='aqua', y=year, m=get_month_folder(date)))
    
                    terra_img = rio.open(image_dataloc + \
                                        'fram_strait-{y}/{m}/labeled_raw/{d}.{s}.labeled_raw.250m.tiff'.format(
                        d=date.strftime('%Y%m%d'), s='terra', y=year, m=get_month_folder(date)))
                    
                    aqua_labels = aqua_img.read().squeeze()
                    terra_labels = terra_img.read().squeeze()
        
                    # get grid for images
                    left, bottom, right, top = aqua_img.bounds
                    n, m = aqua_img.shape
                    X = np.linspace(left, right, m)
                    Y = np.linspace(bottom, top, n)
                    
                    # load asi data
                    with xr.open_dataset(asi_dataloc + yf + '/' + file) as ds_asi:
                        asi_sel = ds_asi.sel(x=slice(left, right), y=slice(bottom, top))
                        asi_coarsened = asi_sel.coarsen({'x': 4, 'y': 4}, boundary='trim').mean().interp(
                            {'x': X, 'y': Y}, method='nearest')
                        asi_upsampled = asi_sel.interp({'x': X, 'y': Y}, method='nearest')
                        sic_6_25km = asi_upsampled['z'].fillna(0).data[::-1, ::] # flip to same orientation as Labels
                        sic_25km = asi_coarsened['z'].fillna(0).data[::-1, ::] # flip to same orientation as Labels
                        for satellite, labels in zip(['aqua', 'terra'], [aqua_labels, terra_labels]):
                            props = pd.DataFrame(
                                regionprops_table(labels, sic_6_25km,
                                                  properties=['label', 'intensity_mean', 'intensity_std', 'area',
                                                              'perimeter', 'bbox', 'centroid'])
                            )
                            labels_expanded = skimage.segmentation.expand_labels(labels, 8)
                            props_expanded = pd.DataFrame(
                                regionprops_table(labels_expanded, sic_6_25km,
                                                  properties=['label', 'intensity_mean', 'intensity_std', 'area',
                                                              'perimeter', 'bbox', 'centroid'])
                            )
                           
                            props_coarsened = pd.DataFrame(
                                regionprops_table(labels, sic_25km,
                                                    properties=['label', 'intensity_mean', 'intensity_std'])
                            )
                            
                            rename = {'centroid-0': 'row_pixel',
                                      'centroid-1': 'col_pixel',
                                      'bbox-0': 'bbox_min_row',
                                      'bbox-1': 'bbox_min_col',
                                      'bbox-2': 'bbox_max_row',
                                      'bbox-3': 'bbox_max_col',
                                      'intensity_mean': 'asi_sic_mean',
                                      'intensity_std': 'asi_sic_std'}
                            props.rename(rename, axis=1, inplace=True)
                            props_expanded.rename(rename, axis=1, inplace=True)
                            props_coarsened.rename(rename, axis=1, inplace=True)
        
                            props_expanded = props_expanded.add_suffix('_expanded')
                            props_coarsened = props_coarsened.add_suffix('_coarsened')
                            props_expanded.rename({'label_expanded': 'label'}, axis=1, inplace=True)
                            props_coarsened.rename({'label_coarsened': 'label'}, axis=1, inplace=True)
                            props_merged = props.merge(props_expanded, left_on='label', right_on='label').merge(
                                props_coarsened,  left_on='label', right_on='label')
        
                            props_merged['satellite'] = satellite
                            props_merged['date'] = date.strftime('%Y%m%d')
        
                            year_data.append(props_merged.copy())
                    
                            del props, props_expanded, props_merged, props_coarsened
                    del aqua_labels, terra_labels
                except:
                    print(date)
                    
                
        df_all = pd.concat(year_data).reset_index(drop=True)
        df_all.to_csv('../data/floe_tracker/ift_floe_property_tables_asi/ift_raw_floe_properties_asi_{y}.csv'.format(y=year))