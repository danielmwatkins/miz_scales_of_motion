"""Generate set of polygons fulfilling the criteria for minimum angle, then calculate the strain rates and total deformation.
"""
import argparse
import itertools
import numpy as np
import os
import pandas as pd
import proplot as pplt
import pyproj

# parser = argparse.ArgumentParser()
# parser.add_argument("year", help="Year between 2003 and 2020. Only calculates polygons for this year.",
#                     type=int)
# args = parser.parse_args()
# year = args.year

# Using the ft_with_nsidc csv file that has all the years of data in it, as well as having
# additional info. Can replace if the data is updated.
ift_data = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
ift_data['datetime'] = pd.to_datetime(ift_data['datetime'])
# ift_data = ift_data.loc[ift_data.datetime.dt.year == year]
ift_data.dropna(subset='u', inplace=True) # Only keep data where velocity was calculated
init_n = len(ift_data)

strict_filter = False

if strict_filter:
    # option to include only data where sea ice concentration and the NSIDC data is available
    sic_sim_check = (ift_data['sea_ice_concentration'] <= 1) & (ift_data['u_nsidc'].notnull())
    ift_data = ift_data.loc[sic_sim_check,:].copy()

print('Filter results: initially len', init_n, 'now len', len(ift_data))
projIn = 'epsg:4326' # WGS 84
projOut = 'epsg:6931' # NSIDC EASE 2.0 (for area calculation)
transformer_laea = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)
x, y = transformer_laea.transform(ift_data['longitude'], ift_data['latitude'])
ift_data['x'] = x
ift_data['y'] = y

# Few steps: make a dataframe with dates as keys, group the dates by year
# and then get a list of dates
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
    
    theta_min = np.min([np.rad2deg(theta1), np.rad2deg(theta2), np.rad2deg(theta3)], axis=0)
    
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

def mean_accel(xcomp, ucomp, area, sign):
    """xcomp and ucomp input should be an N x 3 array. Different gradients will need
    different combinations of x, y, u, and v. In particular:
        dudx = accel(Y, U, A, 1)
        dudy = accel(X, U, A, -1)
        dvdx = accel(Y, V, A, 1)
        dvdy = accel(X, V, A, -1)    
    """

    nr, nc = xcomp.shape
    total = np.zeros(nr)
    for idx in range(nc):
        idx1 = (idx + 1) % nc
        total += (ucomp[:, idx1] + ucomp[:, idx])*(xcomp[:, idx1] - xcomp[:, idx])
    return 1/(2*area) * total * sign


results = {}
variables = ['floe_id',  'latitude', 'longitude', 'x', 'y',
             'u', 'v', 'u_nsidc', 'v_nsidc', 
             'nsidc_sic', 'zeta', 'area_km2', 'edge_dist_km', 'coast_dist_km']

min_angle = 20
data = []
year = 2002
for date in dates:
    if date.year != year:
        print(date.year)
        year += 1

    df = ift_by_date[date]
    if len(df) >= 3:
        all_triangles = pd.DataFrame([list(x) for x in itertools.combinations(df.index, 3)],
                                     columns=['floe1', 'floe2', 'floe3'])
    
        df_triangles = pd.concat([df.loc[all_triangles['floe' + ii],variables].add_suffix(ii).reset_index(drop=True)
                                  for ii in ['1', '2', '3']], axis=1)
        area_km2 = np.round(area(
            x_coords = [df_triangles['x1'], df_triangles['x2'], df_triangles['x3']],
            y_coords = [df_triangles['y1'], df_triangles['y2'], df_triangles['y3']]) / 1e6, -1)
    
        # if the area is negative, we swap the places of floe1 and floe3
        idx_swap = area_km2 < 0
        floe1 = all_triangles['floe1'].copy()
        floe3 = all_triangles['floe3'].copy()
        all_triangles.loc[idx_swap, 'floe1'] = floe3.loc[idx_swap]
        all_triangles.loc[idx_swap, 'floe3'] = floe1.loc[idx_swap]
        
        df_triangles = pd.concat([df.loc[all_triangles['floe' + ii],variables].add_suffix(ii).reset_index(drop=True)
                                  for ii in ['1', '2', '3']], axis=1)
        area_km2 = np.round(area(
            x_coords = [df_triangles['x1'], df_triangles['x2'], df_triangles['x3']],
            y_coords = [df_triangles['y1'], df_triangles['y2'], df_triangles['y3']]) / 1e6, -1)
    
        if area_km2.min() < 0:
            print('Negative area on ', date)
    
        df_triangles['triangle_area_km2'] = area_km2
        df_triangles['min_angle_deg'] = check_shape(
            x_coords = [df_triangles['x1'], df_triangles['x2'], df_triangles['x3']],
            y_coords = [df_triangles['y1'], df_triangles['y2'], df_triangles['y3']])
        df_triangles = df_triangles.loc[df_triangles['min_angle_deg'] >= min_angle]
        df_triangles['datetime'] = date
        if len(df_triangles) > 0:
            data.append(df_triangles)
all_results = pd.concat(data).reset_index(drop=True)
all_results['month'] = all_results['datetime'].dt.month
all_results['year'] = all_results['datetime'].dt.year    

xcoords = np.array([all_results['x1'], all_results['x2'], all_results['x3']]).T
ycoords = np.array([all_results['y1'], all_results['y2'], all_results['y3']]).T
ucoords = np.array([all_results['u1'], all_results['u2'], all_results['u3']]).T
vcoords = np.array([all_results['v1'], all_results['v2'], all_results['v3']]).T

area_m2 = all_results['triangle_area_km2'].values*1e6 # convert back to meters. This area was calculated with LAEA
dudx = mean_accel(ycoords, ucoords, area_m2, 1)
dudy = mean_accel(xcoords, ucoords, area_m2, -1)
dvdx = mean_accel(ycoords, vcoords, area_m2, 1)
dvdy = mean_accel(xcoords, vcoords, area_m2, -1)

all_results['divergence'] = dudx + dvdy #div
all_results['vorticity'] = dvdx - dudy #vor
all_results['pure_shear'] = dudy + dvdx #pure
all_results['normal_shear'] = dudx - dvdy #normal
all_results['total_deformation'] = 0.5*np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2) #epsilon_ii = total deformation

# for c in results.columns:
#     if c[0] != 'f':
#         results[c] = np.round(results[c], 5)
#     if c == 'area':
#         results[c] = np.round(results[c], 0)

# results.to_csv('../data/deformation/polygons_' + str(year) + '.csv')