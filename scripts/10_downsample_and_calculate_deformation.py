import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import Point, Polygon
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

# Define logarithmically spaced bins
logbins = np.logspace(np.log(10), np.log(300), base=np.e, num=10)


# Helper function for deformation
def mean_accel(xcomp, ucomp, area, sign):
    """xcomp and ucomp input should be an N x 3 array. Different gradients will need
    different combinations of x, y, u, and v. In particular:
        dudx = accel(Y, U, A, 1)
        dudy = accel(X, U, A, -1)
        dvdx = accel(Y, V, A, 1)
        dvdy = accel(X, V, A, -1)
    xcomp should be in polar stereographic components.    
    """
    nr, nc = xcomp.shape
    total = np.zeros(nr)
    for idx in range(nc):
        idx1 = (idx + 1) % nc
        total += (ucomp[:, idx1] + ucomp[:, idx])*(xcomp[:, idx1] - xcomp[:, idx])
    return 1/(2*area) * total * sign

def polygon_area_uncertainty(X, Y, position_uncertainty):
    """Compute the area uncertainty following Dierking et al. 2020"""
    N = X.shape[1]
    S = 0
    for i in range(N):
        # the modulus here makes the calculation wrap around to the beginning
        # could adjust the other codes to do this too
        S += (X[:, (i+1) % N] - X[:, (i-1) % N])**2 +  (Y[:, (i+1) % N] - Y[:, (i-1) % N])**2
    return np.sqrt(0.25*position_uncertainty**2*S)

def gradvel_uncertainty(X, Y, U, V, A, position_uncertainty, time_delta, vel_var='u', x_var='x'):
    """Equation 19 from Dierking et al. 2020 assuming uncertainty in position is same in both x and y.
    Also assuming that there is no uncertainty in time. Default returns standard deviation
    uncertainty for dudx.
    """
    sigma_A = polygon_area_uncertainty(X, Y, position_uncertainty)
    sigma_X = position_uncertainty
    
    # velocity uncertainty
    if vel_var=='u':
        u = U.copy()
    else:
        u = V.copy()
    if x_var == 'x':
        # If you want dudx, integrate over Y
        x = Y.copy()
    else:
        x = X.copy()
    
    sigma_U = 2*sigma_X**2/time_delta**2
    
    
    N = X.shape[1]
    S1, S2, S3 = 0, 0, 0
    for i in range(N):
        # the modulus here makes the calculation wrap around to the beginning
        # could adjust the other codes to do this too
        S1 += (u[:, (i+1) % N] + u[:, (i-1) % N])**2 * (x[:, (i+1) % N] - x[:, (i-1) % N])**2
        S2 += (x[:, (i+1) % N] - x[:, (i-1) % N])**2
        S3 += (u[:, (i+1) % N] + u[:, (i-1) % N])**2
        
    var_ux = sigma_A**2/(4*A**4)*S1 + \
             sigma_U**2/(4*A**2)*S2 + \
             sigma_X**2/(4*A**2)*S3       
    
    return np.sqrt(var_ux)



# Loading data
data = []
years = []
all_data = []
for year in range(2003, 2021):
    results = pd.read_csv('../data/deformation/polygons_' + str(year) + '.csv')
    results = results.dropna(subset=['u1', 'u2', 'u3'])
    results.rename({'Unnamed: 0': 'datetime', 'Unnamed: 1': 'triangle_number'}, axis=1, inplace=True)
    results['datetime'] = pd.to_datetime(results['datetime'].values)
    
    results['L'] = np.sqrt(results['polygon_area'])
    results['log_bin'] = np.digitize(results['L'], logbins)
    all_data.append(results)
    years.append(year)
    data.append(results.groupby('log_bin').count()['datetime'].values)
df_info = pd.DataFrame(data, index=years)

all_results = pd.concat(all_data, axis=0)
all_results.reset_index(drop=True, inplace=True)

# get xcoords
projIn = 'epsg:4326' # WGS 84 Ellipsoid
projOut = 'epsg:3413' # NSIDC North Polar Stereographic
transformer_ps = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

for idx in range(1, 4):
    x, y = transformer_ps.transform(all_results['lon' + str(idx)], all_results['lat' + str(idx)])
    all_results['x' + str(idx)] = x
    all_results['y' + str(idx)] = y

all_results['month'] = all_results['datetime'].dt.month
all_results['year'] = all_results['datetime'].dt.year

# calculate deformation
xcoords = np.array([all_results['x1'], all_results['x2'], all_results['x3']]).T
ycoords = np.array([all_results['y1'], all_results['y2'], all_results['y3']]).T
ucoords = np.array([all_results['u1'], all_results['u2'], all_results['u3']]).T
vcoords = np.array([all_results['v1'], all_results['v2'], all_results['v3']]).T

area = all_results['polygon_area'].values*1e6 # convert back to meters. 

dudx = mean_accel(ycoords, ucoords, area, 1)
dudy = mean_accel(xcoords, ucoords, area, -1)
dvdx = mean_accel(ycoords, vcoords, area, 1)
dvdy = mean_accel(xcoords, vcoords, area, -1)

# from Lopez-Acosta et al. 2019
position_uncertainty = 255
time_delta = 24*60*60

sigma_A = polygon_area_uncertainty(xcoords, ycoords, position_uncertainty)
sigma_dudx = gradvel_uncertainty(xcoords, ycoords, ucoords, vcoords, area,
                                 position_uncertainty, time_delta, vel_var='u', x_var='x')
sigma_dvdx = gradvel_uncertainty(xcoords, ycoords, ucoords, vcoords, area,
                                 position_uncertainty, time_delta, vel_var='v', x_var='x')
sigma_dudy = gradvel_uncertainty(xcoords, ycoords, ucoords, vcoords, area,
                                 position_uncertainty, time_delta, vel_var='u', x_var='y')
sigma_dvdy = gradvel_uncertainty(xcoords, ycoords, ucoords, vcoords, area,
                                 position_uncertainty, time_delta, vel_var='v', x_var='y')

sigma_div = np.sqrt(sigma_dudx**2 + sigma_dvdy**2)
sigma_vrt = np.sqrt(sigma_dvdx**2 + sigma_dudy**2)
sigma_shr = np.sqrt((normal_shear/maximum_shear_strain_rate)**2 * (sigma_dudx**2 + sigma_dvdy**2) + \
                    (pure_shear/maximum_shear_strain_rate)**2 * (sigma_dudy**2 + sigma_dvdx**2))
sigma_tot = np.sqrt((maximum_shear_strain_rate/total_deformation)**2 * sigma_shr**2 + \
                    (divergence/total_deformation)**2 * sigma_vrt**2)



all_results['divergence'] = dudx + dvdy #div
all_results['vorticity'] = dvdx - dudy #vor
all_results['pure_shear'] = dudy + dvdx #pure
all_results['normal_shear'] = dudx - dvdy #normal
all_results['maximum_shear_strain_rate'] = 0.5*np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
all_results['total_deformation'] = np.sqrt(all_results['divergence']**2 + \
                                           2*all_results['maximum_shear_strain_rate']**2)

all_results['uncertainty_area'] = sigma_A,
all_results['uncertainty_divergence'] = sigma_div,
all_results['uncertainty_vorticity'] = sigma_vrt,
all_results['uncertainty_shear'] = sigma_shr,
all_results['uncertainty_total'] = sigma_tot,



# Unique floes sample
all_results['unique_floes_sample'] = False
for date, date_group in all_results.groupby('datetime'):
    for bin_number, bin_group in date_group.groupby('log_bin'):
        used = []
        # set a different random state for each date and bin number, so it's reproducible but not 
        # sorting things the same way each day
        random_state = pd.to_datetime(date).year*1000 + pd.to_datetime(date).dayofyear + bin_number
        
        for row, data in bin_group.sample(frac=1, random_state=random_state).iterrows():
            if (data.floe1 in used) | (data.floe2 in used) | (data.floe3 in used):
                pass
            else:
                used.append(data.floe1)
                used.append(data.floe2)
                used.append(data.floe3)
                all_results.loc[row, 'unique_floes_sample'] = True

# Non-overlapping polygons
# 1. Randomly select first triangle and make a Shapely polygon. Set 'no_overlap_sample' to True
# 2. Check whether the intersection with the next triangle is nonzero. If so, add sample to list
all_results['no_overlap_sample'] = False 
tol = 1e-10

for date, date_group in all_results.groupby('datetime'):
    if date.day % 10 == 0:
        print(date)
    for bin_number, df_bin in date_group.groupby('log_bin'):
        polygons = []
        random_state = pd.to_datetime(date).year*1000 + pd.to_datetime(date).dayofyear + bin_number
        if len(df_bin) > 0:
            polygons = []
            row_keys = []
            # Shuffle the order then make a list of polygons and rows
            for row, data in df_bin.sample(frac=1, replace=False, random_state=random_state).iterrows():
                polygons.append(Polygon([Point(x, y) for x, y in zip([data.x1, data.x2, data.x3],
                                                            [data.y1, data.y2, data.y3])]))
                row_keys.append(row)
            
        non_overlapping = []
        non_overlapping_rows = []
        for r, n, p in zip(row_keys[:-1], range(1, len(polygons)-1), polygons[:-1]):
            if not any(p.intersection(g).area > tol for g in non_overlapping):
                non_overlapping.append(p)
                non_overlapping_rows.append(r)  
        all_results.loc[non_overlapping_rows, 'no_overlap_sample'] = True

columns = ['datetime', 'triangle_number', 'floe1', 'floe2', 'floe3', 'u1', 'u2',
           'u3', 'v1', 'v2', 'v3', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3',
           'area_km21', 'area_km22', 'area_km23', 'zeta1', 'zeta2', 'zeta3',
           'edge_dist_km1', 'edge_dist_km2', 'edge_dist_km3', 'coast_dist_km1',
           'coast_dist_km2', 'coast_dist_km3', 'polygon_area', 'min_angle', 'L', 'log_bin',
           'divergence', 'vorticity', 'pure_shear', 'normal_shear', 'maximum_shear_strain_rate', 'total_deformation',
           'no_overlap_sample', 'unique_floes_sample']

all_results['sampled'] = all_results['no_overlap_sample'] | all_results['unique_floes_sample']
all_results.loc[all_results.sampled, columns].to_csv('../data/deformation/sampled_results.csv')