"""Code to parse the IFT matlab files and produce CSV files that have all the information in place.

The parse_raw requires the data in <dataloc> to have a folder for each year. In each folder, there is an <input> and a <tracked> folder. The <tracked> folder contains order.mat, delta_t.mat, x3.mat, y3.mat, and many others. The data used here was in Rosalinda's folder with supporting material for the absolute and relative dispersion paper.  The time data file is produced by merging SOIT output with a file linking the day of year and the original file name, which contained the information on which satellite the image comes from. 

Files needed in "unparsed" folder for each year:
- /tracked/x3.mat
- /tracked/y3.mat
- /tracked/prop.mat
- /time_data.csv
- /THETA_aqua_before.mat
- /THETA_terra_before.mat


The 2020 images were stretched in the vertical, so a separate conversion step is used there vs the prior years.
Rotation rates are added in the second step, since they are estimated over multiple images and averaged across the two satellite data sources.

Floe shapes are located in the FLOE_LIBRARY.mat files initially. Parsed and indexed floe shapes are saved...


"""
import numpy as np
import os
import pandas as pd
import pyproj
import rasterio
from scipy.interpolate import interp1d
from scipy.io import loadmat

saveloc = '../data/floe_tracker/parsed/'
dataloc = '../data/floe_tracker/unparsed/' 
ref_image = 'NE_Greenland.2017100.terra.250m.tif'

# Copy of the parser that has different names so it can read the x_fixed files
def parser_ift(year, dataloc, ref_image):
    """Read the separate matlab and csv files with info on x, y, and datetime,
    and join into one dataframe"""

    # Load x and y pixel locations from matlab files
    Xi = loadmat('{d}/{y}/tracked/x3.mat'.format(d=dataloc, y=year))['x2']
    Yi = loadmat('{d}/{y}/tracked/y3.mat'.format(d=dataloc, y=year))['y2']
    
    
    # Get the datetimes associated with each image
    info_df = pd.read_csv('{d}/{y}/time_data.csv'.format(
        d=dataloc, y=year), index_col=0)
    info_df['datetime'] = pd.to_datetime(info_df['SOIT time'])
    info_df.set_index('matlab_index', inplace=True)
    
    # Create a dataframe so that each row is given a unique floe id and each 
    # column is a datetime. No need to sort the data at this point - sorting by
    # datetime later will suffice.
    new_index = lambda n: pd.Index([str(year) + '_' + str(idx + 1).zfill(5)
                                     for idx in range(n)], name='floe_id')

    floe_ids = new_index(Xi.shape[0])
    col_names = info_df.index
    df_x = pd.DataFrame(Xi, columns=col_names, index=floe_ids)
    df_y = pd.DataFrame(Yi, columns=col_names, index=floe_ids)
    
    # Rotation rates are calculated individually for each satellite.
    # Later, the best estimate for each day will be calculated through a merging procedure
    # based on rotation measurement availability and an adjustment based on time between images.
    # Rotation in column i is the rotation from day i to day i+1.
    theta_aqua = loadmat('{d}/{y}/tracked/THETA_aqua_before.mat'.format(
        d=dataloc, y=year))['THETA_aqua']
    theta_terra = loadmat('{d}/{y}/tracked/THETA_terra_before.mat'.format(
        d=dataloc, y=year))['THETA_terra']
    
    # making dataframe with floe labels
    # operating under the assumption that the rows and columns are in the same order
    n_floes_aqua, n_days_aqua = theta_aqua.shape
    n_floes_terra, n_days_terra = theta_terra.shape
    aqua_idx = info_df.loc[info_df.satellite == 'aqua'].iloc[:n_days_aqua].index
    terra_idx = info_df.loc[info_df.satellite == 'terra'].iloc[:n_days_terra].index
    
    n_floes = np.min([n_floes_aqua, Xi.shape[0]])
    n_days = np.min([n_days_aqua, len(aqua_idx)])
    df_theta_aqua = pd.DataFrame(theta_aqua[:n_floes,:n_days],
                                 columns=aqua_idx[:n_days], index=floe_ids[:n_floes])
    n_floes = np.min([n_floes_terra, Xi.shape[0]])
    n_days = np.min([n_days_aqua, len(terra_idx)])
    df_theta_terra = pd.DataFrame(theta_terra[:n_floes,:n_days],
                                  columns=terra_idx[:n_days], index=floe_ids[:n_floes])  

    #### Join into a melted dataframe ####    
    # Reduce data size by "melting" the data frame: produces a dataframe with
    # columns for floe id, datetime, x_pixel, and y_pixel, dropping points where
    # there is no data. Then X and Y are merged into a single dataframe.
    tol = 1e-16
    df_x = df_x.where(np.abs(df_x) > tol).melt(
        ignore_index=False).reset_index().rename({'value': 'x_pixel'}, axis=1).dropna()
    df_y = df_y.where(np.abs(df_y) > tol).melt(
        ignore_index=False).reset_index().rename({'value': 'y_pixel'}, axis=1).dropna()

    df = df_x.merge(df_y, left_on=['floe_id', 'matlab_index'], right_on=['floe_id', 'matlab_index'])

    df_theta_aqua = df_theta_aqua.where(np.abs(df_theta_aqua) > tol).melt(
        ignore_index=False).reset_index().rename({'value': 'theta_aqua'}, axis=1).dropna()
    df_theta_terra = df_theta_terra.where(np.abs(df_theta_terra) > tol).melt(
        ignore_index=False).reset_index().rename({'value': 'theta_terra'}, axis=1).dropna()
    
    
    df = df.merge(df_theta_aqua,
                  left_on=['floe_id', 'matlab_index'],
                  right_on=['floe_id', 'matlab_index'], how='outer')
    df = df.merge(df_theta_terra,
                  left_on=['floe_id', 'matlab_index'],
                  right_on=['floe_id', 'matlab_index'], how='outer')
    
    # add info on the satellite for each source
    df['datetime'] = info_df['datetime'].loc[df['matlab_index'].values].values
    df['satellite'] = info_df['satellite'].loc[df['matlab_index'].values].values

    n_init = len(df)
    df = df.dropna(subset=['x_pixel'])
    if n_init > len(df):
        print('Missing x_pixel data dropped')
    
    # Convert to stereographic using information from the reference image
    # For polar stereographic images like this, it is a simple affine transformation
    ref_raster = rasterio.open('{d}/{r}'.format(d=dataloc, r=ref_image))
    if year != 2020:
        x_stere, y_stere = ref_raster.xy(row=df['y_pixel'], col=df['x_pixel'])
        df['x_stere'] = x_stere
        df['y_stere'] = y_stere

    else:
        if year == 2020:
            # The images from 2020 reference a different image and
            # are stretched in the y direction. This applies a linear correction.
            # We first shift from pixels to stereographic
            info_region_pixel_scale_x = 200.2979
            info_region_pixel_scale_y = 208.3310
            x_cropped = 2.0080e+05
            y_cropped = -3.1754e+05
            df['x_stere'] = x_cropped + df['x_pixel'] * info_region_pixel_scale_x
            df['y_stere'] = y_cropped - df['y_pixel'] * info_region_pixel_scale_y
        
            # Then we stretch the image vertically
            left=200704
            bottom=-2009088.0
            right=1093632.0
            top=-317440.0
            adjustment = 63.8e3
            A = ((top - bottom) + adjustment)/(top - bottom)
            B = top * (1 - A)
            df['y_stere'] = A*df['y_stere'] + B

    # Extract the EPSG code for the coordinate reference system from the reference image
    # then use PyProj to convert to longitude and latitude
    source_crs = 'epsg:' + str(ref_raster.crs.to_epsg())
    to_crs = 'WGS84'
    ps2ll = pyproj.Transformer.from_crs(source_crs, to_crs, always_xy=True)
    lon, lat = ps2ll.transform(df['x_stere'], df['y_stere'])

    df['longitude'] = np.round(lon, 5)
    df['latitude'] = np.round(lat, 5)

    # Load the properties matrix, and match tracked floes to floe properties
    # by checking the index of the closest match for the x and the y positions.
    # Keep only the shape properties I think we will use
    props = loadmat('{d}/{y}/tracked/prop.mat'.format(d=dataloc, y=year))['properties'][0]
    df_by_date = {date: group.set_index('floe_id') for date, group in df.groupby('datetime')}
    
    # check for duplicates
    for date in df_by_date:
        if np.any(df_by_date[date].duplicated()):
            df_by_date[date] = df_by_date[date].loc[
                    ~df_by_date[date].index.duplicated(
                        keep='first')].copy()
            
    matched_props = {}
    all_props = {}
    for idx, date in enumerate(info_df['datetime']):
        if date in df_by_date:
            p_df = pd.DataFrame(props[idx],
                columns=['area','perimeter','major_axis','minor_axis',
                         'orientation','x_pixel', 'y_pixel','convex_area','solidity',
                         'bbox1', 'bbox2', 'bbox3','bbox4'])

            p_df['floe_id'] = np.nan
            p_df['datetime'] = date
            floe_id_list = df_by_date[date].index
            for floe_id in floe_id_list:            
                px_idx = np.abs(p_df.x_pixel.values - \
                                df_by_date[date].loc[floe_id, 'x_pixel']).argmin()
                py_idx = np.abs(p_df.y_pixel.values - \
                                df_by_date[date].loc[floe_id, 'y_pixel']).argmin()
                
                if px_idx == py_idx:
                    # add threshold to make sure that it's a match
        
                    dx = p_df.loc[px_idx, 'x_pixel'] - df_by_date[date].loc[floe_id, 'x_pixel']
                    dy = p_df.loc[py_idx, 'y_pixel'] - df_by_date[date].loc[floe_id, 'y_pixel']
                    
                    if np.max((dx, dy)) > 1e-3:
                        print('Larger than threshold')
                    
                    p_df.loc[px_idx, 'floe_id'] = floe_id
            all_props[idx] = p_df.copy()
            matched_props[idx] = p_df.dropna().loc[:,
                    ['floe_id', 'datetime', 'area',
                     'perimeter', 'major_axis', 'minor_axis',
                     'orientation']]

    
    # Finally, link the matched properties and the trajectories
    # and sort by floe_id and by datetime
    df_props = pd.concat(matched_props).reset_index(drop=True)

    # Add correct position data to all_props matrix
    df_all_props = pd.concat(all_props).reset_index(drop=True)
    if year == 2020:
        # The images from 2020 reference a different image and
        # are stretched in the y direction. This applies a linear correction.
        # We first shift from pixels to stereographic
        info_region_pixel_scale_x = 200.2979
        info_region_pixel_scale_y = 208.3310
        x_cropped = 2.0080e+05
        y_cropped = -3.1754e+05
        df_all_props['x_stere'] = x_cropped + df_all_props['x_pixel'] * info_region_pixel_scale_x
        df_all_props['y_stere'] = y_cropped - df_all_props['y_pixel'] * info_region_pixel_scale_y
    
        # Then we stretch the image vertically
        left=200704
        bottom=-2009088.0
        right=1093632.0
        top=-317440.0
        adjustment = 63.8e3
        A = ((top - bottom) + adjustment)/(top - bottom)
        B = top * (1 - A)
        df_all_props['y_stere'] = A*df_all_props['y_stere'] + B
    else:
        x_stere, y_stere = ref_raster.xy(row=df_all_props['y_pixel'], col=df_all_props['x_pixel'])
        df_all_props['x_stere'] = x_stere
        df_all_props['y_stere'] = y_stere

    lon, lat = ps2ll.transform(df_all_props['x_stere'], df_all_props['y_stere'])
    df_all_props['longitude'] = np.round(lon, 5)
    df_all_props['latitude'] = np.round(lat, 5)
    
    
    df_merged = df.merge(df_props, left_on=['floe_id', 'datetime'], right_on=['floe_id', 'datetime'])
    df_merged.sort_values(['floe_id', 'datetime']).reset_index(drop=True)
    return df_merged, df_all_props

for year in range(2003, 2021):
    print(year)
    df, props = parser_ift(year=year,
            dataloc=dataloc,
            ref_image='NE_Greenland.2017100.terra.250m.tif')
    
    n_missing = len(df.loc[df.x_pixel.isnull()])
    if n_missing > 0:
        print('Warning: Missing position data for', n_missing, 'floes')

    df.to_csv(saveloc + 'tracked_floes/ift_tracked_floes_{y}.csv'.format(y=year))
    props.to_csv(saveloc + 'all_floes/ift_floe_properties_{y}.csv'.format(y=year))