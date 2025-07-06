import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import proplot as pplt
import pyproj
import scipy.stats as stats
from scipy.interpolate import interp2d
import sys
import warnings

sys.path.append('../scripts/')
from drifter import compute_along_across_components


pplt.rc.reso = 'med'
pplt.rc['cartopy.circular'] = False

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

df_ift = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
df_ift['datetime'] = pd.to_datetime(df_ift['datetime'].values)

# Calculations
# Length scale bins need area adjustment
df_ift['area_adj_km2'] = (np.sqrt(df_ift.area) + 8)**2*.25*.25 # 6 pixel shift minimizes error against manual

edge_bins = np.arange(0, 800, 25)
df_ift['edge_bin'] = np.digitize(df_ift.edge_dist_km, bins=edge_bins)

length_bins = np.arange(0, 50, 2)
df_ift['length_scale'] = df_ift['area_adj_km2']**0.5
df_ift['length_bin'] = np.digitize(df_ift.length_scale, bins=length_bins)


sim_ualong = pd.read_csv('../data/simulation/u_along.csv', index_col=0)
sim_uacross = pd.read_csv('../data/simulation/u_across.csv', index_col=0)
sim_std_dev = pd.read_csv('../data/simulation/stdev.csv', index_col=0)

##### Calculate Velocity Mean and Comparison with NSIDC ######
min_x = df_ift.x_stere.min()
max_x = df_ift.x_stere.max()
min_y = df_ift.y_stere.min()
max_y = df_ift.y_stere.max()

x_bins = np.arange(min_x, max_x, 25e3)
y_bins = np.arange(min_y, max_y, 25e3)
xc = 0.5*(x_bins[1:] + x_bins[:-1])
yc = 0.5*(y_bins[1:] + y_bins[:-1])
X, Y = np.meshgrid(xc, yc)

# Select data from 25 km bins for IFT and for NSIDC to compute monthly means
crs0 = pyproj.CRS('WGS84')
crs1 = pyproj.CRS('epsg:3413')
transformer_pstere = pyproj.Transformer.from_crs(crs1, crs_to=crs0, always_xy=True)
 
lon_grid, lat_grid = transformer_pstere.transform(np.ravel(X), np.ravel(Y))
lon_grid = np.reshape(lon_grid, X.shape)
lat_grid = np.reshape(lat_grid, Y.shape)

u_data = {}
v_data = {}
hist = {}
diffs_mean = {}
diffs_u = {}
diffs_v = {}

for month in [4, 5, 6]:
    u_data[month] = {}
    v_data[month] = {}
    hist[month] = {}
    for label, suffix in zip(['IFT', 'NSIDC'], ['', '_nsidc']):
        sel = (df_ift.datetime.dt.month == month) & (df_ift['u'].notnull())
        x = df_ift.x_stere
        y = df_ift.y_stere
        u = df_ift['u' + suffix]
        v = df_ift['v' + suffix]
        
        hist2d = np.histogram2d(df_ift.loc[sel, :].dropna(subset=['u', 'u_nsidc'], axis=0)['x_stere'],
                       df_ift.loc[sel, :].dropna(subset=['u', 'u_nsidc'], axis=0)['y_stere'],
                      bins=[x_bins, y_bins])
        df_hist = pd.DataFrame(hist2d[0], index=xc, columns=yc)
        hist[month] = df_hist
        
        u_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
            x[sel], y[sel], values=u[sel], statistic='mean', 
            bins=[x_bins, y_bins])
        v_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
            x[sel], y[sel], values=v[sel], statistic='mean', 
            bins=[x_bins, y_bins])

        # Rotation from Earth coordinates to the north polar stereographic grid for display
        U_nps = u_mean.T * np.sin(np.deg2rad(lon_grid + 45)) + v_mean.T * np.cos(np.deg2rad(lon_grid + 45))
        V_nps = v_mean.T * np.cos(np.deg2rad(lon_grid + 45)) - u_mean.T * np.sin(np.deg2rad(lon_grid + 45))
        u_data[month][label] = pd.DataFrame(U_nps.T, index=xc, columns=yc)
        v_data[month][label] = pd.DataFrame(V_nps.T, index=xc, columns=yc)

    # Calculate the mean of the differences from vector components
    # and calculate the L2 norm of the difference 
    
    diff_u = df_ift['u'] - df_ift['u_nsidc']
    diff_v = df_ift['v'] - df_ift['v_nsidc']
    diff_norm = np.sqrt(diff_u**2 + diff_v**2)
    dn, xedges, yedges, binnumber = stats.binned_statistic_2d(
                x[sel], y[sel], values=diff_norm[sel], statistic='mean', 
                bins=[x_bins, y_bins])
    diffs_mean[month] = pd.DataFrame(dn, index=xc, columns=yc)
    du, xedges, yedges, binnumber = stats.binned_statistic_2d(
                x[sel], y[sel], values=diff_u[sel], statistic='mean', 
                bins=[x_bins, y_bins])
    dv, xedges, yedges, binnumber = stats.binned_statistic_2d(
                x[sel], y[sel], values=diff_v[sel], statistic='mean', 
                bins=[x_bins, y_bins])
    
    # Rotation from Earth coordinates to the north polar stereographic grid for display
    U_nps = du.T * np.sin(np.deg2rad(lon_grid + 45)) + dv.T * np.cos(np.deg2rad(lon_grid + 45))
    V_nps = dv.T * np.cos(np.deg2rad(lon_grid + 45)) - du.T * np.sin(np.deg2rad(lon_grid + 45))
    diffs_u[month] = pd.DataFrame(U_nps.T, index=xc, columns=yc)
    diffs_v[month] = pd.DataFrame(V_nps.T, index=xc, columns=yc)

    print('Month', month, 'mean difference', diffs_mean[month].mean().mean().round(2))
crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
fig, axs = pplt.subplots(width=8, proj='npstere',
                         proj_kw={'lon_0': -45}, ncols=3, nrows=2, share=False)

for ax in axs:
    ax.set_extent([0.4e6, 1.1e6, -2.15e6, -0.7e6], crs=crs)  
    ax.format(land=True, coast=True, 
           landzorder=0, landcolor='k', facecolor='w')
for ax in axs[:, -1]:
    # scale arrow
    # ax.text(0.91e6, -2.13e6, '200 km', color='k')     
    ax.quiver(0.85e6 + 60e3, -2.0e6, 0.2, 0, color='k', scale=1, width=1/200, label='', transform=crs)
    ax.text(0.91e6, -1.975e6, '20 cm/s', transform=crs)
    # spatial scale
    ax.plot([0.85e6, 0.85e6 + 200e3], [-2.1e6, -2.1e6], lw=4, color='k', zorder=3, transform=crs)
    ax.text(0.91e6, -2.07e6, '200 km', color='k', transform=crs)
    

for col, month in zip([0, 1, 2], [4, 5, 6]):

    idx_data = hist[month] > 30
    
    # Plot count of IFT observations
    c0 = axs[0, col].pcolor(lon_grid, lat_grid, hist[month].where(idx_data).T.values, vmin=0, vmax=100,
           transform=ccrs.PlateCarree(), cmap='blues', extend='max')
    
    axs[0, col].quiver(lon_grid, lat_grid, u_data[month]['IFT'].where(idx_data).T.values, v_data[month]['IFT'].where(idx_data).T.values,
               transform=ccrs.PlateCarree(), color='r', scale=1, width=1/250, label='IFT')
    axs[0, col].quiver(lon_grid, lat_grid, u_data[month]['NSIDC'].where(idx_data).T.values, v_data[month]['NSIDC'].where(idx_data).T.values,
               transform=ccrs.PlateCarree(), color='k', scale=1, width=1/200, label='NSIDC')
    
    axs[0, 0].legend(loc='ll', ncols=1, alpha=1, lw=2)
    
    c1 = axs[1, col].pcolor(lon_grid, lat_grid, diffs_mean[month].where(idx_data).T.values, vmin=0, vmax=0.3,
           transform=ccrs.PlateCarree(), cmap='reds', extend='max', N=7)
    axs[1, col].quiver(lon_grid, lat_grid, diffs_u[month].where(idx_data).T.values, diffs_v[month].where(idx_data).T.values,
               transform=ccrs.PlateCarree(), color='k', scale=1, width=1/300)

    
        
axs[0, col].colorbar(c0, loc='r', shrink=0.85, label='Count', labelsize=11)
axs[1, col].colorbar(c1, loc='r', shrink=0.85, label='Vector Difference (m/s)', labelsize=11)
axs.format(leftlabels = ['Mean Drift','Difference'],
           toplabels=['April', 'May', 'June'], fontsize=12, abc=True)
fig.save('../figures/fig11_mean_drift.png', dpi=300)
fig.save('../figures/fig11_mean_drift.pdf')







##### Figure 12: Velocity Distribution

# Compute anomaly relative to the NSIDC data
tau = '5D'
comp = df_ift.copy()
comp['u'] = comp['u'] - comp['u_nsidc']
comp['v'] = comp['v'] - comp['v_nsidc']
df_comp = compute_along_across_components(comp, uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')

length_bins = np.arange(0, 50, 2)

df_edge = pd.concat({'sigma_ut': df_comp[['edge_bin', 'U_fluctuating']].groupby('edge_bin').std(),
                      'sigma_ul': df_comp[['edge_bin', 'U_along']].groupby('edge_bin').std(),
              'n': df_comp[['edge_bin', 'U_fluctuating']].groupby('edge_bin').count(),
             'd': df_comp[['edge_bin', 'edge_dist_km']].groupby('edge_bin').mean()}, axis=1)
df_edge.columns = pd.Index(['sigma_ut', 'sigma_ul', 'n', 'd'])

df_lscale = pd.concat({'sigma_ut': df_comp[['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df_comp[['length_bin', 'U_along']].groupby('length_bin').std(),
              'n': df_comp[['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
             'L': df_comp[['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_lscale.columns = pd.Index(['sigma_ut', 'sigma_ul', 'n', 'L'])

R = sim_std_dev['u_along (m/s)'].index
L = np.sqrt(np.pi)*R

normal_dist = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-1/2 * x**2)

## Directional anomalies
sigma = 1
normal_dist = lambda x: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2 * (x/sigma)**2)

fig, axs = pplt.subplots(height=6, nrows=2, ncols=3, share=False)
x_bins = np.linspace(-0.6, 0.6, 51)
for tau, ls in zip(['5D', '15D', '31D'], ['-', '--', ':']):
    df_rot_ift = compute_along_across_components(df_ift.copy(), uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')
    df_rot_nsidc = compute_along_across_components(df_ift.drop(['u', 'v'], axis=1).rename({'u_nsidc': 'u', 'v_nsidc': 'v'}, axis=1), uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')
    for data, color, product in zip([df_rot_ift, df_rot_nsidc], ['r', 'k'], ['IFT', 'NSIDC']):
    
        u = data['U_along']
        v = data['U_fluctuating']
        u_pdf, _ = np.histogram(u, bins=x_bins, density=True)
        v_pdf, _ = np.histogram(v, bins=x_bins, density=True)
        x_center = 1/2*(x_bins[1:] + x_bins[:-1])

        label = product
        if tau != '5D':
            label = ''
            
        axs[0, 0].plot(x_center, u_pdf, color=color, label=label, ls=ls)
        axs[1, 0].plot(x_center, v_pdf, color=color, label=label, ls=ls)
    comp = df_ift.copy()
    comp['u'] = comp['u'] - comp['u_nsidc']
    comp['v'] = comp['v'] - comp['v_nsidc']
    df_comp = compute_along_across_components(comp, uvar='u', vvar='v',
                                        umean='u' + tau + '_nsidc',
                                        vmean='u' + tau + '_nsidc')
    u = df_comp['U_along']
    v = df_comp['U_fluctuating']
    u_pdf, _ = np.histogram(u, bins=x_bins, density=True)
    v_pdf, _ = np.histogram(v, bins=x_bins, density=True)
    x_center = 1/2*(x_bins[1:] + x_bins[:-1])

    label = 'Diff'
    if tau != '5D':
        label = ''
        
    axs[0, 0].plot(x_center, u_pdf, color='tab:blue', label=label, ls=ls, lw=1)
    axs[1, 0].plot(x_center, v_pdf, color='tab:blue', label=label, ls=ls, lw=1)

axs[0, 0].legend(loc='ul', ncols=1)
# axs[1, 0].legend(loc='ul', ncols=1)

h = [axs[0,0].plot([],[], lw=1, color='gray', ls=ls) for ls in ['-', '--', ':']]
# axs[0,0].legend(h, ['5D', '15D', '31D'], loc='ur', ncols=1)
axs[1,0].legend(h, ['5D', '15D', '31D'], loc='ul', ncols=1)

axs[0,0].format(title='', ylabel='Probability')
axs[1,0].format(title='', ylabel='Probability')
axs[0,0].format(xlabel='$u_L$ (m/s)', yscale='log', xlim=(-0.4, 0.4), ylim=(0.1, 13), abc=True)
axs[1,0].format(xlabel='$u_T$ (m/s)', yscale='log', xlim=(-0.4, 0.4), ylim=(0.1, 13), abc=True)


#### Fluctuating Velocity Distributions
for ax, symb, var in zip([axs[0, 1], axs[1, 1]], ['d', '+'], ['U_along', 'U_fluctuating']):
    ustd = df_comp[var].std()
    u = df_comp[var]
    print(tau, var, np.round(ustd*100, 3), 'cm/s')   
    pdf, x_bins = np.histogram(u/ustd, bins=np.linspace(-7, 7, 151), density=True)
    x_center = 1/2*(x_bins[1:] + x_bins[:-1])
    if 'along' in var:
        label_pdf = 'N(0, 1)'
    else:
        label_pdf = ''

    ax.scatter(x_center, pdf, marker=symb, zorder=5, color='tab:blue', label='Observed') 
    # ax.plot(x_center, normal_dist(x_center), marker='',
    #         lw=0.5, color='k', ls='--', label=label_pdf, zorder=1)


    train = df_comp.dropna(subset=var).sample(1000, replace=False)
    test = df_comp.dropna(subset=var)
    test = test.loc[[x for x in test.index if x not in train.index]]
    abs_u_train = np.abs(train[var])/np.std(train[var])
    abs_u_test = np.abs(test[var])/np.std(test[var])
    exp_loc, exp_scale = stats.expon.fit(abs_u_train, floc=0)
    print('Fitted exponential dist. scale param:', np.round(exp_scale, 2))
    expon_dist = stats.expon(loc=0, scale=exp_scale).pdf
    normal_dist = stats.norm(loc=0, scale=1).pdf
    
    ax.plot(x_center, normal_dist(x_center), marker='',
            lw=1, color='k', ls='--', label='N(0, 1)', zorder=10)
    ax.plot(x_center, 0.5*expon_dist(np.abs(x_center)), marker='',
            lw=1, color='k', ls=':', label='Exp({s})'.format(s=np.round(exp_scale, 2)), zorder=10)

# The normalization in np.histogram is not the same as the normalization in
# Minki's matlab routine, to bring them to the same convention, we need to divide by
# the bin spacing. 
dx = np.diff(sim_uacross.index)[0]
   
for col in sim_ualong.columns:
    if col == sim_ualong.columns[0]:
        label='Simulated'
    else:
        label=''
    axs[0, 1].scatter(sim_ualong.index, sim_ualong.loc[:, col]/dx, color='gray',
                m='d', alpha=0.25, label=label, zorder=0)
    axs[1, 1].scatter(sim_uacross.index, sim_uacross.loc[:, col]/dx, color='gray',
                alpha=0.25, m='+',  label=label, zorder=0)
    
axs[0, 1].format(title='', xlabel='$u\'_L$ (m/s)/$\sigma_{u\'_L}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')
axs[1, 1].format(title='', xlabel='$u\'_T$ (m/s)/$\sigma_{u\'_T}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')

axs[0, 1].legend(ncols=1)
axs[1, 1].legend(ncols=1)

# Length scale dependence, for length scales with at least 300 observations
idx = df_lscale.n > 300
ax = axs[0, 2]
ax.scatter(df_lscale.loc[idx, 'L'].values,
        df_lscale.loc[idx, 'sigma_ul'].values,
        marker='d', label='', color='tab:blue')
ax.scatter(df_lscale.loc[idx, 'L'].values,
        df_lscale.loc[idx, 'sigma_ut'].values,
        marker='+', label='', color='tab:blue')
ax.scatter(L, sim_std_dev['u_along (m/s)'],
           marker='d', label='', color='gray')
ax.scatter(L, sim_std_dev['u_across (m/s)'],
           marker='+', label='', color='gray')
ax.format(xlabel='Length scale (km)', ylabel='$\sigma_{u\'}$', title='', ylim=(0, 0.12), xlim=(0, 60), yscale='linear')
# ax.plot([], 

# Edge distance, for edge bins with at least 300 observations
idx = df_edge.n > 300
ax = axs[1, 2]
ax.scatter(df_edge.loc[idx, 'd'].values,
               df_edge.loc[idx, 'sigma_ul'].values, marker='d', color='tab:blue', label='Longitudinal')
ax.scatter(df_edge.loc[idx, 'd'].values,
               df_edge.loc[idx, 'sigma_ut'].values, marker='+', color='tab:blue', label='Transverse')
ax.format(ylim=(0, 0.12), ytickminor=False, xtickminor=False, yscale='linear', xlim=(0, 400),
         ylabel='$\sigma_{u\'}$', xlabel='Distance to ice edge (km)', fontsize=12)
ax.legend(loc='ur', ncols=1)

l = ['Observed', 'Simulated', 'Longitudinal', 'Transverse']
h = [ax.plot([],[], marker=m, color=c, lw=0) for m, c in zip(['o', 'o', 'd', '+'],
                                                           ['tab:blue', 'gray', 'k', 'k'])]
axs[0, 2].legend(h, l, ncols=1)

fig.format(fontsize=12)
fig.save('../figures/fig12_velocity_obs_sim.pdf', dpi=300)
fig.save('../figures/fig12_velocity_obs_sim.png', dpi=300)
