import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import ultraplot as pplt
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
                     'mean_ut': np.abs(df_comp[['edge_bin', 'U_fluctuating']]).groupby('edge_bin').mean(),
                      'mean_ul': np.abs(df_comp[['edge_bin', 'U_along']]).groupby('edge_bin').mean(),
              'n': df_comp[['edge_bin', 'U_fluctuating']].groupby('edge_bin').count(),
             'd': df_comp[['edge_bin', 'edge_dist_km']].groupby('edge_bin').mean()}, axis=1)
df_edge.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul',  'n', 'd'])

df_lscale = pd.concat({'sigma_ut': df_comp[['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df_comp[['length_bin', 'U_along']].groupby('length_bin').std(),
                      'mean_ut': np.abs(df_comp[['edge_bin', 'U_fluctuating']]).groupby('edge_bin').mean(),
                      'mean_ul': np.abs(df_comp[['edge_bin', 'U_along']]).groupby('edge_bin').mean(),
              'n': df_comp[['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
             'L': df_comp[['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_lscale.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul', 'n', 'L'])

normal_dist = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-1/2 * x**2)

## Directional anomalies
sigma = 1
normal_dist = lambda x: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2 * (x/sigma)**2)

fig, axs = pplt.subplots(height=3, nrows=1, ncols=3, share=False)
x_bins = np.linspace(-0.6, 0.6, 51)
tau = '5D'

comp = df_ift.copy()
comp['u'] = comp['u'] - comp['u5D_nsidc']
comp['v'] = comp['v'] - comp['v5D_nsidc']
df_comp = compute_along_across_components(comp, uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')

#### Fluctuating Velocity Distributions
ax = axs[0]
for symb, c, var in zip(['d', '+'], ['tab:blue', 'tab:orange'], ['U_along', 'U_fluctuating']):
    ustd = df_comp[var].std()
    u = df_comp[var]
    print(tau, var, np.round(ustd*100, 3), 'cm/s')   
    pdf, x_bins = np.histogram(u/ustd, bins=np.linspace(-7, 7, 151), density=True)
    x_center = 1/2*(x_bins[1:] + x_bins[:-1])
    if 'along' in var:
        label_pdf = 'N(0, 1)'
    else:
        label_pdf = ''

    ax.scatter(x_center, pdf, marker=symb, zorder=5, color=c, label='') 

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
    
axs[0].format(title='', xlabel='$u\'_L$ (m/s)/$\sigma_{u\'_L}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')

axs[0].legend(ncols=1)


# Length scale dependence, for length scales with at least 300 observations
idx = df_lscale.n > 300
ax = axs[1]

ax.plot(df_lscale.loc[idx, 'L'].values,
               df_lscale.loc[idx, 'mean_ul'].values, marker='d', color='tab:blue', label='Longitudinal')

ax.plot(df_lscale.loc[idx, 'L'].values,
               df_lscale.loc[idx, 'sigma_ul'].values, marker='d', ls='--', lw=1, color='tab:blue', label='Longitudinal')

ax.plot(df_lscale.loc[idx, 'L'].values,
               df_lscale.loc[idx, 'mean_ut'].values, marker='.', color='tab:orange', label='Transverse')
ax.plot(df_lscale.loc[idx, 'L'].values,
               df_lscale.loc[idx, 'sigma_ut'].values, marker='.', ls='--', lw=1, color='tab:orange', label='Transverse')



ax.format(xlabel='Length scale (km)', ylabel='m/s', title='', ylim=(0, 0.1), xlim=(5, 30), yscale='linear')

# Edge distance, for edge bins with at least 300 observations
idx = df_edge.n > 300
ax = axs[2]
ax.plot(df_edge.loc[idx, 'd'].values,
               df_edge.loc[idx, 'mean_ul'].values, marker='d', color='tab:blue', label='Longitudinal')

ax.plot(df_edge.loc[idx, 'd'].values,
               df_edge.loc[idx, 'sigma_ul'].values, marker='d', ls='--', lw=1, color='tab:blue', label='Longitudinal')

ax.plot(df_edge.loc[idx, 'd'].values,
               df_edge.loc[idx, 'mean_ut'].values, marker='.', color='tab:orange', label='Transverse')
ax.plot(df_edge.loc[idx, 'd'].values,
               df_edge.loc[idx, 'sigma_ut'].values, marker='.', ls='--', lw=1, color='tab:orange', label='Transverse')


ax.format(ylim=(0, 0.1), ytickminor=False, xtickminor=False, yscale='linear', xlim=(0, 400),
         ylabel='$m/s$', xlabel='Distance to ice edge (km)', fontsize=12)

l = ['Longitudinal', 'Transverse', 'Mean', 'St. Dev.']
h = [ax.plot([],[], marker=m, color=c, lw=lw, ls=ls) for m, c, lw, ls in zip(['d', 'o', '', ''],
                                                           ['tab:blue', 'tab:orange', 'k', 'k'],
                                                                    [0, 0, 1, 2], ['-', '-', '-', '--'])]
axs[1].legend(h, l, ncols=1, loc='ll')
axs[2].legend(h, l, ncols=1, loc='ll')

fig.format(fontsize=12)

fig.save('../figures/figXX_simpler_velocity_dist.png', dpi=300)