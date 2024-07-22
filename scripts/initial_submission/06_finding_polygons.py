"""Code to identify and extract info for polygons used in the deformation calculation.
Area is calculated in LAEA, while the minimum interior angle is calculated in north polar stereographic.
Stereographic projections are conformal (angle preserving) on a sphere, so for the earth ellipsoid, the
projection is not perfectly conformal, however, it should be close enough to conformal to use it for ruling 
out overly-stretched polygons. All polygons with minimum angle greater than 20 are kept. The order of coordinates
is clockwise, checked by using signed area calculations.
"""
import argparse
import itertools
import numpy as np
import os
import pandas as pd
import proplot as pplt
import pyproj


parser = argparse.ArgumentParser()
parser.add_argument("year", help="Year between 2003 and 2020. Only calculates polygons for this year.",
                    type=int)
args = parser.parse_args()
year = args.year

# Using the ft_with_nsidc csv file that has all the years of data in it, as well as having
# additional info. Can replace if the data is updated.
ift_data = pd.read_csv('../data/floe_tracker/ift_with_nsidc.csv', index_col=0)
ift_data['datetime'] = pd.to_datetime(ift_data['datetime'])
ift_data = ift_data.loc[ift_data.datetime.dt.year == year]
ift_data = ift_data.loc[ift_data.qc_flag==0].copy()
init_n = len(ift_data)

# Apply filters
# speed_check = (ift_data['speed'] <= 1.5) & (ift_data['speed'] > 0.01)
# circ_check = (4*np.pi*ift_data['area']/ift_data['perimeter']**2) >= 0.6
sic_sim_check = (ift_data['sea_ice_concentration'] <= 1) & (ift_data['u_nsidc'].notnull())
ift_data = ift_data.loc[sic_sim_check,:].copy()

print('Filter results: initially len', init_n, 'now len', len(ift_data))

projIn = 'epsg:4326' # WGS 84 Ellipsoid
projOut = 'epsg:3413' # NSIDC North Polar Stereographic
transformer_ps = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

projOut = 'epsg:6931' # NSIDC EASE 2.0 (for area calculation)
transformer_laea = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)
x, y = transformer_laea.transform(ift_data['longitude'], ift_data['latitude'])
ift_data['x_laea'] = x
ift_data['y_laea'] = y

# Few steps: make a dataframe with dates as keys, group the dates by year
# and then get a list of 
ift_by_date = {date: group for date, group in ift_data.groupby('datetime')}
dates = list(ift_by_date.keys())

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



results = {}
for date in dates:
    df = ift_by_date[date]
    all_triangles = [list(x) for x in itertools.combinations(df.index, 3)]
    keep = []
    for coords in all_triangles:
        min_angle = check_shape(df.x_stere[coords].values, df.y_stere[coords].values, min_angle=20)
        if min_angle >= 20:
            A = area(df.x_laea[coords].values, df.y_laea[coords].values)
            if A < 1:
                coords = [coords[0], coords[2], coords[1]]
                A = np.abs(A)
            
            vals = []
            for attr in ['floe_id', 'u', 'v', 'latitude', 'longitude']:
                vals += [df.loc[cc, attr] for cc in coords]
            vals += [A/1e6, min_angle]
            keep.append(vals)

    results[date] = pd.DataFrame(keep, columns = ['floe1', 'floe2', 'floe3', 
                                                  'u1', 'u2', 'u3', 'v1', 'v2', 'v3',
                                                  'lat1', 'lat2', 'lat3', 'lon1', 'lon2', 'lon3',
                                                  'area', 'min_angle'])

# round down the precision to 5 for everything numeric
# area is in km2

results = pd.concat(results)
for c in results.columns:
    if c[0] != 'f':
        results[c] = np.round(results[c], 5)
    if c == 'area':
        results[c] = np.round(results[c], 0)

results.to_csv('../data/deformation/polygons_' + str(year) + '.csv')