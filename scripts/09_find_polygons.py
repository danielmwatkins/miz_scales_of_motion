"""Code to identify and extract info for polygons used in the deformation calculation.
Area is calculated in LAEA, while the minimum interior angle is calculated in north polar stereographic.
Stereographic projections are conformal (angle preserving) on a sphere, so for the earth ellipsoid, the
projection is not perfectly conformal, however, it should be close enough to conformal to use it for ruling 
out overly-stretched polygons. All polygons with minimum angle greater than 20 are kept, and the minimum angle
is calculated in order to test sensitivity of results to this choice. The order of coordinates
is clockwise, checked by using signed area calculations.
"""

import itertools
import numpy as np
import os
import pandas as pd
import proplot as pplt
import pyproj

def net_displacement_meters(floe_df):
    """Calculates net pixel displacement for trajectory"""
    delta_x = floe_df['x_stere'].values[0] - floe_df['x_stere'].values[-1]
    delta_y = floe_df['y_stere'].values[0] - floe_df['y_stere'].values[-1]
    return np.sqrt(delta_x**2 + delta_y**2)

def estimated_max_speed(floe_df):
    """Calculates distance traversed in units of pixels"""
    delta_x = floe_df['x_stere'] - floe_df['x_stere'].shift(-1)
    delta_y = floe_df['y_stere'] - floe_df['y_stere'].shift(-1)
    dt = floe_df['datetime'] - floe_df['datetime'].shift(-1)
    return np.round(np.abs(np.sqrt(delta_x**2 + delta_y**2)/dt.dt.total_seconds()).max(), 3)
    
ift_data = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv')
ift_data['datetime'] = pd.to_datetime(ift_data['datetime'])

# Require a minimum of at least 1 pixel total displacement
ift_data = ift_data.groupby('floe_id').filter(lambda x: net_displacement_meters(x) > 500)

# Max speed has to be less than 1 m/s and greater than 0.05 m/s
ift_data = ift_data.groupby('floe_id').filter(lambda x: (estimated_max_speed(x) < 1) & \
            (estimated_max_speed(x) > 0.05))
init_n = len(ift_data)

# Apply filters
sic_sim_check = (ift_data['nsidc_sic'] <= 1) & (ift_data['u_nsidc'].notnull())
ift_data = ift_data.loc[sic_sim_check,:].copy()

print('Filter results: initially len', init_n, 'now len', len(ift_data))

def check_shape(x_coords, y_coords, min_angle=30, return_val='min_angle'):
    """Assumes a triangle. Calculates interior angles, then checks if the minimum
    interior angle is at least <min_angle>. Can return either True/False, minimum angle,
    or all angles. For true false, use return_val='flag', for angle, use 'min_angle',
    for all, use 'verbose'. Flag is True if the minimum angle is too small. """
    a = np.sqrt((x_coords[0] - x_coords[1])**2 + (y_coords[0] - y_coords[1])**2)
    b = np.sqrt((x_coords[1] - x_coords[2])**2 + (y_coords[1] - y_coords[2])**2)
    c = np.sqrt((x_coords[2] - x_coords[0])**2 + (y_coords[2] - y_coords[0])**2)
    theta1 = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    theta2 = np.arccos((b**2 + c**2 - a**2)/(2*b*c))
    theta3 = np.pi - theta1 - theta2

    theta_min = np.min([np.rad2deg(theta1), np.rad2deg(theta2), np.rad2deg(theta3)])
    
    if return_val=='verbose':
        return np.rad2deg(theta1), np.rad2deg(theta2), np.rad2deg(theta3)
    elif return_val=='flag':
        return theta_min < min_angle
    else:
        return theta_min

def area(x_coords, y_coords):
    """X and Y are vectors with the coordinates of the polygon nodes. Returns sign of the
    area - should be positive, otherwise swap the coordinate order."""
    s2 = 0.
    N = len(x_coords)
    s1 = x_coords[N-1]*y_coords[0] - x_coords[0]*y_coords[N-1]
    for i in range(N - 1):
        s2 += x_coords[i]*y_coords[i+1] - y_coords[i]*x_coords[i+1]
    return (s2 + s1)/2

# Make dataframe with dates as keys, then for each date loop through all positive
# triangles and save those with minimum angle larger than 20 degrees
for year, ift_df in ift_data.groupby(ift_data.datetime.dt.year):
    ift_by_date = {date: group for date, group in ift_df.groupby('datetime')}
    dates = list(ift_by_date.keys())
    
    results = {}
    for date in dates:
        df = ift_by_date[date].dropna(subset='u')
        all_triangles = [list(x) for x in itertools.combinations(df.index, 3)]
        keep = []
        for coords in all_triangles:
            min_angle = check_shape(df.x_stere[coords].values,
                                    df.y_stere[coords].values, min_angle=20)
            if min_angle >= 20:
                A = area(df.x_stere[coords].values, df.y_stere[coords].values)
                if A < 1:
                    coords = [coords[0], coords[2], coords[1]]
                    A = np.abs(A)
                
                vals = []
                for attr in ['floe_id', 'u', 'v', 'latitude', 'longitude',
                             'area_km2', 'zeta', 'edge_dist_km', 'coast_dist_km']:
                    vals += [df.loc[cc, attr] for cc in coords]
                vals += [A/1e6, min_angle]
                keep.append(vals)
    
        results[date] = pd.DataFrame(keep, columns = ['floe1', 'floe2', 'floe3', 
                                                      'u1', 'u2', 'u3',
                                                      'v1', 'v2', 'v3',
                                                      'lat1', 'lat2', 'lat3',
                                                      'lon1', 'lon2', 'lon3',
                                                      'area_km21', 'area_km22', 'area_km23',
                                                      'zeta1', 'zeta2', 'zeta3',
                                                      'edge_dist_km1', 'edge_dist_km2', 'edge_dist_km3',
                                                      'coast_dist_km1', 'coast_dist_km2', 'coast_dist_km3',
                                                      'polygon_area',
                                                      'min_angle'])
    

    # Reduce number of significant figures
    results = pd.concat(results)
    for c in results.columns:
        if c[0] != 'f':
            results[c] = np.round(results[c], 5)
        if c == 'area':
            results[c] = np.round(results[c], 0)
    
    results.to_csv('../data/deformation/polygons_' + str(year) + '.csv')