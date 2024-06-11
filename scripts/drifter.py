# TBD
# Set up tests for each function
# Set up shape-check and order-check

import pandas as pd
import numpy as np
import pyproj

    
def compute_velocity(buoy_df, date_index=True, rotate_uv=False, method='c', xvar='x', yvar='y'):
    """Computes buoy velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto stereographic x/y prior
    to calculating velocity. Rotate_uv moves the velocity into east/west. Velocity
    calculations are done on the provided time index. Results will not necessarily 
    be reliable if the time index is irregular. With centered differences, values
    near endpoints are calculated as forward or backward differences.
    
    Options for method
    forward (f): forward difference, one time step
    backward (b): backward difference, one time step
    centered (c): 3-point centered difference
    forward_backward (fb): minimum of the forward and backward differences
    
    TBD: Add option to fit smooth function and calculate derivate from values of the smooth
    function, e.g. by fitting a spline.
    TBD: Make simple test to make sure methods are called correctly
    TBD: Harmonize the API for specifying date column
    TBD: use something like **args to collect optional inputs
    TBD: Improve angle method so it doesn't throw errors, and only compute heading if well defined
    """
    buoy_df = buoy_df.copy()
    
    if date_index:
        date = pd.Series(pd.to_datetime(buoy_df.index.values), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df.date)
        
    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)
    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    
    if xvar not in buoy_df.columns:
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3413' # NSIDC North Polar Stereographic
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_df[xvar] = x
        buoy_df[yvar] = y
    
    if method in ['f', 'forward']:
        dt = (date.shift(-1) - date).dt.total_seconds().values
        dxdt = (buoy_df[xvar].shift(-1) - buoy_df[xvar])/dt
        dydt = (buoy_df[yvar].shift(-1) - buoy_df[yvar])/dt

    elif method in ['b', 'backward']:
        dt = (date - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df[xvar] - buoy_df[xvar].shift(1))/dt
        dydt = (buoy_df[yvar] - buoy_df[yvar].shift(1))/dt

    elif method in ['c', 'fb', 'centered', 'forward_backward']:
        fwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='forward')
        bwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='backward')

        fwd_dxdt, fwd_dydt = fwd_df['u'], fwd_df['v']
        bwd_dxdt, bwd_dydt = bwd_df['u'], bwd_df['v']
        
        if method in ['c', 'centered']:
            dt = (date.shift(-1) - date.shift(1)).dt.total_seconds()
            dxdt = (buoy_df[xvar].shift(-1) - buoy_df[xvar].shift(1))/dt
            dydt = (buoy_df[yvar].shift(-1) - buoy_df[yvar].shift(1))/dt
        else:
            dxdt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dxdt, 'b':bwd_dxdt})).min(axis=1)
            dydt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dydt, 'b':bwd_dydt})).min(axis=1)

        dxdt.loc[fwd_endpoint] = fwd_dxdt.loc[fwd_endpoint]
        dxdt.loc[bwd_endpoint] = bwd_dxdt.loc[bwd_endpoint]
        dydt.loc[fwd_endpoint] = fwd_dydt.loc[fwd_endpoint]
        dydt.loc[bwd_endpoint] = bwd_dydt.loc[bwd_endpoint]
    
    if rotate_uv:
        # Unit vectors
        buoy_df['Nx'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * -buoy_df[xvar]
        buoy_df['Ny'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * -buoy_df[yvar]
        buoy_df['Ex'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * -buoy_df[yvar]
        buoy_df['Ey'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * buoy_df[xvar]

        buoy_df['u'] = buoy_df['Ex'] * dxdt + buoy_df['Ey'] * dydt
        buoy_df['v'] = buoy_df['Nx'] * dxdt + buoy_df['Ny'] * dydt

        # Calculate angle, then change to 360
        heading = np.degrees(np.angle(buoy_df.u.values + 1j*buoy_df.v.values))
        heading = (heading + 360) % 360
        
        # Shift to direction from north instead of direction from east
        heading = 90 - heading
        heading = (heading + 360) % 360
        buoy_df['bearing'] = heading
        buoy_df['speed'] = np.sqrt(buoy_df['u']**2 + buoy_df['v']**2)
        buoy_df.drop(['Nx', 'Ny', 'Ex', 'Ey'], axis=1, inplace=True)
        
    else:
        buoy_df['u'] = dxdt
        buoy_df['v'] = dydt            
        buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)    

    return buoy_df
    
def compute_absolute_dispersion(vel_varname, data, max_length='30D', step_size=3600):
    """Computes the absolute dispersion for buoys in data. Data need
    to be aligned to a common time step. Assumes the start time is time 0,
    and will use data up to time 0 + max_length. Step size in seconds.
    """
    dt = pd.to_timedelta(max_length)
    vel_df = pd.DataFrame({b: data[b][vel_varname].loc[
                    slice(data[b].index[0], data[b].index[0] + dt)]
                           for b in data})
    # calculate integral
    x_df = pd.DataFrame({b: np.cumsum(vel_df[b]*step_size) for b in vel_df.columns})
    x_df = (x_df.T - x_df.iloc[0,:]).T # remove initial position
    N = len(x_df.columns)
    return 1/(N-1)*(x_df**2).sum(axis=1)

def compute_along_across_components(buoy_df, uvar='u', vvar='v', umean='u_mean', vmean='v_mean'):
    """Project the velocity into along-track and across track components."""

    ub = buoy_df[uvar]
    us = buoy_df[umean]
    vb = buoy_df[vvar]
    vs = buoy_df[vmean]

    scale = (ub*us + vb*vs)/(us**2 + vs**2)
    buoy_df[uvar + '_along'] = scale*us
    buoy_df[vvar + '_along'] = scale*vs

    buoy_df[uvar + '_across'] = ub - buoy_df[uvar + '_along']
    buoy_df[vvar + '_across'] = vb - buoy_df[vvar + '_along']
    
    sign = np.sign(buoy_df[umean]*buoy_df[vvar + '_across'] - buoy_df[vmean]*buoy_df[uvar + '_across'])
    # fluctuating component of velocity
    buoy_df['U_fluctuating'] = sign * np.sqrt(
         buoy_df[uvar + '_across']**2 + \
         buoy_df[vvar + '_across']**2)
    
    # along-track component of velocity
    buoy_df['U_along'] = sign * np.sqrt(
         buoy_df[uvar + '_along']**2 + \
         buoy_df[vvar + '_along']**2)
    
    return buoy_df


def compute_strain_rate_components(buoys, data, position_uncertainty, time_delta='1H'):
    """Compute the four components of strain rate for each
    date in data. Assumes velocity has already been calculated.
    Expects "data" to be a dictionary with a dataframe for each
    of the buoys in the list "buoys". The dataframes in "data"
    should have columns "u", "v", "longitude", "latitude". Buoys should
    be listed so that the polygon linking them is in counterclockwise order.

    Output: dataframe with columns 'divergence', 'vorticity',
             'pure_shear', 'normal_shear', 'maximum_shear_strain_rate',
             'area', 'shape_flag'

    Position uncertainty is expected to be in standard deviation form, so
    the units should be meters.

    Time delta is the length of time used in the calculation of velocity. Needs to be in
    format understandable by pandas to_timedelta(), e.g. '1H' for one hour.
    """

    def polygon_area(X, Y):
        """Compute area of polygon as a sum. Should use LAEA not PS here"""
        sumvar = 0.
        N = len(X)        
        for i in range(N):
            sumvar += X[i]*Y[(i+1) % N] - Y[i]*X[(i+1) % N]
        return sumvar*0.5

    def polygon_area_uncertainty(X, Y, position_uncertainty):
        """Compute the area uncertainty following Dierking et al. 2020"""
        N = len(X)
        S = 0
        for i in range(N):
            # the modulus here makes the calculation wrap around to the beginning
            # could adjust the other codes to do this too
            S += (X[(i+1) % N] - X[(i-1) % N])**2 +  (Y[(i+1) % N] - Y[(i-1) % N])**2
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
        
        
        N = len(X)
        S1, S2, S3 = 0, 0, 0
        for i in range(N):
            # the modulus here makes the calculation wrap around to the beginning
            # could adjust the other codes to do this too
            S1 += (u[(i+1) % N] + u[(i-1) % N])**2 * (x[(i+1) % N] - x[(i-1) % N])**2
            S2 += (x[(i+1) % N] - x[(i-1) % N])**2
            S3 += (u[(i+1) % N] + u[(i-1) % N])**2
            
        var_ux = sigma_A**2/(4*A**4)*S1 + \
                 sigma_U**2/(4*A**2)*S2 + \
                 sigma_X**2/(4*A**2)*S3       
        
        return np.sqrt(var_ux)

    
    def accel(X, U, A, sign):
        """Computes spatial derivative of velocity for 
        deformation."""
        N = len(X)
        sumvar = 0
        for i in range(N):
            sumvar += (U[(i+1) % N] + U[i])*(X[(i+1) % N] - X[i])
        return 1/(2*A) * sumvar * sign

    lon_data = pd.DataFrame({b: data[b]['longitude'] for b in buoys})
    lat_data = pd.DataFrame({b: data[b]['latitude'] for b in buoys})
    time_delta = pd.to_timedelta(time_delta).total_seconds()
    
    # Polar stereographic for velocity-based component
    projIn = 'epsg:4326' # WGS 84 Ellipsoid
    projOut = 'epsg:3413' # NSIDC North Polar Stereographic
    transformer_ps = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

    projOut = 'epsg:6931' # NSIDC EASE 2.0 (for area calculation)
    transformer_laea = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)
    
    X_data = lon_data * np.nan
    Y_data = lon_data * np.nan
    XA_data = lon_data * np.nan
    YA_data = lon_data * np.nan
    U_data = lon_data * np.nan
    V_data = lon_data * np.nan    
    
    for buoy in X_data.columns:
        lon = lon_data[buoy].values
        lat = lat_data[buoy].values

        x, y = transformer_ps.transform(lon, lat)
        X_data[buoy] = x
        Y_data[buoy] = y
        
        xa, ya = transformer_laea.transform(lon, lat)
        XA_data[buoy] = xa
        YA_data[buoy] = ya
        
        buoy_df = pd.DataFrame({'longitude': lon,
                                'latitude': lat,
                                'x': x,
                                'y': y}, index=X_data.index)
        buoy_df = compute_velocity(buoy_df)
        U_data[buoy] = buoy_df['u']
        V_data[buoy] = buoy_df['v']
    
    X = X_data.T.values
    Y = Y_data.T.values
    XA = XA_data.T.values
    YA = YA_data.T.values        
    U = U_data.T.values
    V = V_data.T.values

    A = polygon_area(XA, YA)

    # Check order of points
    # Can't handle reversal partway through though
    if np.all(A[~np.isnan(A)] < 0):
        print('Reversing order')
        X = X[::-1,:]
        XA = XA[::-1,:]
        Y = Y[::-1,:]
        YA = YA[::-1,:]
        U = U[::-1,:]
        V = V[::-1,:]
        
    A = polygon_area(XA, YA)

    dudx = accel(Y, U, A, 1)
    dudy = accel(X, U, A, -1)
    dvdx = accel(Y, V, A, 1)
    dvdy = accel(X, V, A, -1)

    divergence = dudx + dvdy
    vorticity = dvdx - dudy
    pure_shear = dudy + dvdx
    normal_shear = dudx - dvdy
    maximum_shear_strain_rate = np.sqrt(pure_shear**2 + normal_shear**2)
    total_deformation = np.sqrt(divergence**2 + maximum_shear_strain_rate**2)


    sigma_A = polygon_area_uncertainty(X, Y, position_uncertainty)
    sigma_dudx = gradvel_uncertainty(X, Y, U, V, A,
                                     position_uncertainty, time_delta, vel_var='u', x_var='x')
    sigma_dvdx = gradvel_uncertainty(X, Y, U, V, A,
                                     position_uncertainty, time_delta, vel_var='v', x_var='x')
    sigma_dudy = gradvel_uncertainty(X, Y, U, V, A,
                                     position_uncertainty, time_delta, vel_var='u', x_var='y')
    sigma_dvdy = gradvel_uncertainty(X, Y, U, V, A,
                                     position_uncertainty, time_delta, vel_var='v', x_var='y')

    sigma_div = np.sqrt(sigma_dudx**2 + sigma_dvdy**2)
    sigma_vrt = np.sqrt(sigma_dvdx**2 + sigma_dudy**2)
    sigma_shr = np.sqrt((normal_shear/maximum_shear_strain_rate)**2 * (sigma_dudx**2 + sigma_dvdy**2) + \
                        (pure_shear/maximum_shear_strain_rate)**2 * (sigma_dudy**2 + sigma_dvdx**2))
    sigma_tot = np.sqrt((maximum_shear_strain_rate/total_deformation)**2 * sigma_shr**2 + \
                        (divergence/total_deformation)**2 * sigma_vrt**2)
        
    results_df = pd.DataFrame(
        {'divergence': divergence,
         'vorticity': vorticity,
         'pure_shear': pure_shear,
         'normal_shear': normal_shear,
         'maximum_shear_strain_rate': maximum_shear_strain_rate,
         'total_deformation': total_deformation,
         'area': A,
         'uncertainty_area': sigma_A,
         'uncertainty_divergence': sigma_div,
         'uncertainty_vorticity': sigma_vrt,
         'uncertainty_shear': sigma_shr,
         'uncertainty_total': sigma_tot,
         'shape_flag': np.sign(A)},
        index=X_data.index)
    
    return results_df