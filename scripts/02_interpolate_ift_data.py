"""Reads raw IFT csv files from data/floe_tracker/parsed/, resamples to daily resolution,
and adds ERA5 winds. Results for individual years are saved in data/floe_tracker/interpolated
and a single dataframe with wind speed, drift speed ratio, and turning angle for all
the years are saved in data/floe_tracker.

The full files are saved within the floe_tracker/interpolated files. For the analysis, we
apply a filter requiring the ice drift speed to be at least 0.02 m/s and less than 1.5 m/s.
The filtered data is saved as a single CSV file in data/floe_tracker/ft_with_wind.csv.
"""

import numpy as np
import pandas as pd
import pyproj 
from metpy.units import units
import metpy.calc as mcalc
import xarray as xr
import sys
# sys.path.append('/Users/dwatkin2/Documents/research/packages/buoy_processing')
import drifter
from drifter.src.analysis import compute_velocity
from scipy.interpolate import interp1d

saveloc = '../data/floe_tracker/interpolated/'
saveloc_single = '../data/floe_tracker/'
dataloc = '../data/floe_tracker/parsed/tracked_floes/'

max_diff = 30 
max_speed = 1.5 # Threshold to include in the standard deviation calculation
max_speed_z = 6 # Threshold for speed z-score
min_speed = 0.005 # Position uncertainty = 266, so 2*sigma / day
min_circ = 0.6
min_length = 250 # meters!

# Location of folder with ERA5 data. I saved the ERA5 data with 
# a file structure of era5_dataloc/YYYY/era5_uvmsl_daily_mean_YYYY-MM-01.nc
# era5_dataloc = '../external_data/era5_daily/'
def path_length(xp, yp):
    """Integrate the path length in pixels. Used for filtering."""
    return np.sum(np.sqrt((xp - xp.shift(1))**2  + (yp - yp.shift(1))**2))

def regrid_floe_tracker(group, datetime_grid, variables=['x_stere', 'y_stere','longitude', 'latitude',
                                                         'area', 'perimeter', 'major_axis', 'minor_axis']):
    """Estimate the location at 12 UTC using linear interpolation. Group should have datetime index.
    """
    group = group.sort_index()
    begin = group.index.min()
    end = group.index.max()

    if len(datetime_grid.loc[slice(begin, end)]) > 1:
        t0 = group.index.round('12H').min()
        t1 = group.index.round('12H').max()
        max_extrap = pd.to_timedelta('2H')
        if np.abs(t0 - begin) < max_extrap:
            begin = t0
        if np.abs(t1 - end) < max_extrap:
            end = t1

        X = group[variables].T #.rolling(
            #'1H', center=True).mean().T.values
        t_new = datetime_grid.loc[slice(begin, end)].values
        t_seconds = group['t'].values
        Xnew = interp1d(t_seconds, X,
                        bounds_error=False,
                        kind='linear', fill_value='extrapolate')(t_new)
        idx = ~np.isnan(Xnew.sum(axis=0))

        df_new = pd.DataFrame(data=np.round(Xnew.T, 5), 
                              columns=variables,
                              index=datetime_grid.loc[slice(begin, end)].index)
        return df_new
    
    else:
        df = pd.DataFrame(data = np.nan, columns=group.columns, index=[begin])
        df.drop(['floe_id', 't'], axis=1, inplace=True)
        return df

def estimate_theta(df):
    """Estimate the angle change between each satellite from the orientation
    column from the properties matrix. Units are degrees."""
    df = df.copy()
    df['theta_terra_est'] = np.nan
    df['theta_aqua_est'] = np.nan
    df_t = df.loc[df.satellite=='terra']
    df_a = df.loc[df.satellite=='aqua']

    

    
    t_theta_est1 = df_t.orientation.shift(-1) - df_t.orientation
    t_theta_est2 = (df_t.orientation % 180).shift(-1) - df_t.orientation
    t_theta_est3 = df_t.orientation.shift(-1) - (df_t.orientation  % 180)
    comp = pd.concat({'t1': t_theta_est1, 't2': t_theta_est2, 't3': t_theta_est3}, axis=1)
    if len(comp) > 0:
        idx = np.abs(comp).idxmin(axis=1).fillna('t1')
        df.loc[comp.index, 'theta_terra_est'] = pd.Series(
            [comp.loc[x, y] for x, y in zip(idx.index, idx.values)], index=comp.index)
    
    a_theta_est1 = df_a.orientation.shift(-1) - df_a.orientation
    a_theta_est2 = (df_a.orientation % 180).shift(-1) - df_a.orientation
    a_theta_est3 = df_a.orientation.shift(-1) - (df_a.orientation  % 180)
    comp = pd.concat({'t1': a_theta_est1, 't2': a_theta_est2, 't3': a_theta_est3}, axis=1)
    if len(comp) > 0:
        idx = np.abs(comp).idxmin(axis=1).fillna('t1')
        df.loc[comp.index, 'theta_aqua_est'] = pd.Series(
            [comp.loc[x, y] for x, y in zip(idx.index, idx.values)], index=comp.index)

    return df
    
def get_daily_angle_estimate(df, interp_df, max_diff):
    """Takes the angle differences from IFT (theta) and from differences in 
    the properties matrix (theta_est), then estimates the angle for the daily
    grid. The angles are converted to zeta by dividing by the time between 
    satellite images. Returns the merged dataset with zeta in units of radians per day.
    """
    df['datetime'] = pd.to_datetime(df['datetime'].values)
    df['delta_time'] = np.nan
    for sat in ['aqua', 'terra']:
        df_sat = df.loc[df.satellite==sat]
        dt = df_sat.datetime.shift(-1) - df_sat.datetime
        df.loc[df_sat.index, 'delta_time'] = dt.dt.total_seconds() / (60*60*24) # Report rates as per day
    
    df['zeta_aqua'] = np.deg2rad(df['theta_aqua']) / df['delta_time']
    df['zeta_terra'] = np.deg2rad(df['theta_terra']) / df['delta_time']
    df['zeta_aqua_est'] = np.deg2rad(df['theta_aqua_est']) / df['delta_time']
    df['zeta_terra_est'] = np.deg2rad(df['theta_terra_est']) / df['delta_time']
    
    # Add the 
    df_aqua = df.loc[df.satellite=='aqua', ['zeta_aqua', 'zeta_aqua_est']].merge(
        interp_df['x_stere'], left_index=True, right_index=True, how='outer').drop('x_stere', axis=1)
    df_terra = df.loc[df.satellite=='terra', ['zeta_terra', 'zeta_terra_est']].merge(
        interp_df['x_stere'], left_index=True, right_index=True, how='outer').drop('x_stere', axis=1)
    df_aqua = df_aqua.interpolate(method='time', limit=1, limit_direction='both').loc[interp_df.index]
    df_terra = df_terra.interpolate(method='time', limit=1, limit_direction='both').loc[interp_df.index]
    df_angles = df_aqua.merge(df_terra, left_index=True, right_index=True)
    
    
    # Fill if one satellite is missing
    df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_aqua'] = df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_terra']
    df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_aqua_est'] = df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_terra_est']
    df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_terra'] = df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_aqua']
    df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_terra_est'] = df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_aqua_est']
    
    df_angles['zeta_diff'] = np.abs(df_angles['zeta_aqua'] - df_angles['zeta_terra'])
    df_angles['zeta_diff_est'] = df_angles['zeta_aqua_est'] - df_angles['zeta_terra_est']
    df_angles['zeta'] = df_angles[['zeta_aqua', 'zeta_terra']].mean(axis=1).where(df_angles['zeta_diff'] < np.deg2rad(max_diff))
    df_angles['zeta_est'] = df_angles[['zeta_aqua_est', 'zeta_terra_est']].mean(axis=1).where(df_angles['zeta_diff_est'] < np.deg2rad(max_diff))
    
    return interp_df.merge(df_angles[['zeta', 'zeta_est']], left_index=True, right_index=True)

####### Apply the functions to the IFT parsed data #########
ft_df_raw = {}
for year in range(2003, 2021):
    df = pd.read_csv(
        dataloc + 'ift_tracked_floes_' + str(year) + '.csv',
        index_col=None).dropna(subset='x_pixel')
    ft_df_raw[year] = df
    
ft_df_raw = pd.concat(ft_df_raw)
ft_df_raw.index.names = ['year', 'd1']
ft_df_raw = ft_df_raw.reset_index().drop(['d1'], axis=1)
ft_df_raw['date'] = pd.to_datetime(ft_df_raw['datetime'].values).round('1min')

print('Initial number of observations:', len(ft_df_raw))
length = ft_df_raw.groupby('floe_id').apply(lambda x: len(x))
print('Initial number of unique floes:', len(length))

floe_tracker_results = {}
for year, year_group in ft_df_raw.groupby(ft_df_raw.date.dt.year):
    ref_time = pd.to_datetime(str(year) + '-01-01 00:00')
    date_grid = pd.date_range(str(year) + '-04-01 00:00', str(year) + '-09-30 00:00', freq='1D')
    date_grid += pd.to_timedelta('12H')
    t_grid = (date_grid - ref_time).total_seconds()
    year_group['t'] = (year_group['date'] - ref_time).dt.total_seconds()
    datetime_grid = pd.Series(t_grid, index=date_grid)
    
    results = {}
    for floe_id, group in year_group.groupby('floe_id'):
        group = group.loc[~group.date.duplicated()].copy()
        df_orig = estimate_theta(group.set_index('date'))
        df_regrid = regrid_floe_tracker(group.set_index('date'), datetime_grid=datetime_grid)
        if np.any(df_regrid.notnull()):
            # massage the angular measurements into place
        
            df_interp = get_daily_angle_estimate(df_orig, df_regrid, max_diff)            
            results[floe_id] = df_interp.copy()
            del df_interp
        del df_orig, df_regrid

    floe_tracker_results[year] = pd.concat(results)
    floe_tracker_results[year].index.names = ['floe_id', 'date']
    floe_tracker_results[year].reset_index(inplace=True)
    floe_tracker_results[year] = floe_tracker_results[year].loc[:,
                                          ['date', 'floe_id', 'x_stere', 'y_stere', 'longitude', 'latitude',
                                           'area', 'perimeter', 'major_axis', 'minor_axis', 'zeta', 'zeta_est']]
    for var in ['x_stere', 'y_stere', 'longitude', 'latitude']:
        floe_tracker_results[year][var] = floe_tracker_results[year][var].round(5)
    # Change to kilometers
    floe_tracker_results[year]['area'] = floe_tracker_results[year]['area']*0.25**2
    floe_tracker_results[year]['perimeter'] = floe_tracker_results[year]['perimeter']*0.25
    floe_tracker_results[year]['major_axis'] = floe_tracker_results[year]['major_axis']*0.25
    floe_tracker_results[year]['minor_axis'] = floe_tracker_results[year]['minor_axis']*0.25
    floe_tracker_results[year] = floe_tracker_results[year].groupby('floe_id', group_keys=False).apply(
        compute_velocity, date_index=False, rotate_uv=True, method='f', xvar='x_stere', yvar='y_stere').dropna(subset='u')

    floe_tracker_results[year].to_csv(saveloc + '/floe_tracker_interp_' + str(year) + '.csv')


ft_df = pd.concat(floe_tracker_results)
ft_df['circularity'] = 4*np.pi * ft_df['area']/(ft_df['perimeter']**2)

# Filters
circ_check = ft_df.circularity <= min_circ
init_filter = ~circ_check & (ft_df.speed <= max_speed)
sigma_speed = ft_df.loc[init_filter, 'speed'].std()

mean_u = ft_df.loc[init_filter, 'u'].mean()
mean_v = ft_df.loc[init_filter, 'v'].mean()

z = np.sqrt((ft_df.u - mean_u)**2 + (ft_df.v - mean_v)**2)/sigma_speed
speed_check = np.abs(z) > 6

too_short = ft_df.groupby('floe_id').filter(lambda x: path_length(x.x_stere, x.y_stere)/len(x) < min_length) # Less than 1 pixel per day
min_speed_check = ft_df.groupby('floe_id').filter(lambda x: np.max(x.speed) < min_speed)
flag = pd.Series(np.zeros(len(ft_df)), index=ft_df.index)
for test in [speed_check, circ_check, too_short.index, min_speed_check.index]:   
    flag[test] += 1                 

ft_df['qc_flag'] = flag
length = ft_df.groupby('floe_id').apply(lambda x: len(x))
length_qc = ft_df.loc[flag==0].groupby('floe_id').apply(lambda x: len(x))
print('Number of distinct floes:', len(length), len(length_qc))
print('Number of velocity estimates:', len(ft_df), len(ft_df.loc[flag==0]))
print('Median trajectory length:', length.median(), length_qc.median())
ft_df.to_csv(saveloc_single + 'ift_with_zeta.csv')

######## TBD ###########
# 1. Floe rotation rates. Merging rules and verification
# 2. Filtering. Don't remove the low speeds - filter later, e.g. with net transport and with circularity filter
# 3. Retain area, perimeter



