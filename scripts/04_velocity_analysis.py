import numpy as np
import pandas as pd
import proplot as pplt
import os
import pyproj
import sys
import warnings

sys.path.append('../scripts/')
from drifter import compute_velocity
from drifter import compute_along_across_components
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

### Load observed trajectories
ift_loc = '../data/floe_tracker/ift_floe_trajectories.csv'
df_ift = pd.read_csv(ift_loc, index_col=0)
df_ift['datetime'] = pd.to_datetime(df_ift['datetime'].values)

### Simulation data
sim_ualong = pd.read_csv('../data/simulation/u_along.csv', index_col=0)
sim_uacross = pd.read_csv('../data/simulation/u_across.csv', index_col=0)
sim_std_dev = pd.read_csv('../data/simulation/stdev.csv', index_col=0)


#### Group the data by the averaging period, so that the names are consistent for the plotting section
results_anomalies = {}
for tau in ['5D', '15D', '31D']:
    data = pd.DataFrame({'datetime': df_ift['datetime'],
                         'floe_id': df_ift['floe_id'],
                         'edge_dist': df_ift['edge_dist_km'],
                         'floe_size': df_ift['area_km2'],
                         'rotation_rate': df_ift['zeta'],
                         'sea_ice_concentration': df_ift['nsidc_sic'],
                         'u': df_ift['u'] - df_ift['u' + tau + '_nsidc'], 
                         'v': df_ift['v'] - df_ift['v' + tau + '_nsidc'],
                         'u_mean': df_ift['u' + tau + '_nsidc'],
                         'v_mean': df_ift['v' + tau + '_nsidc']
                         }).dropna()
    data = compute_along_across_components(data)
    results_anomalies[tau] = data.copy()

def plot_velocity_distribution(data_df, savename='../figures/velocity_histogram.png',
                             vel_comp1='U_along', vel_comp2='U_fluctuating',
                             vc1_name='Longitudinal', vc2_name='Transverse'):
    """Plots the histograms of the specified velocity components (normalized by standard deviation)
    against a Gaussian distribution. All floes in data_df lumped together. Future update could provide some
    sort of weighting."""

    all_u = data_df[vel_comp1].dropna()
    all_v = data_df[vel_comp2].dropna()
    print(len(all_u))
    fig, axs = pplt.subplots(ncols=2, width=6, sharex=False)
    normal_dist = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-1/2 * x**2)
    for ax, data in zip(axs, [all_u, all_v]):
        pdf, x_bins = np.histogram(data/np.std(data), bins=np.linspace(-7, 7, 151), density=True)
        x_center = 1/2*(x_bins[1:] + x_bins[:-1])
        print(np.std(data))
        ax.plot(x_center, pdf, marker='.', lw=0, color='b') 
        ax.plot(x_center, normal_dist(x_center), marker='', lw=0.5, color='k', ls='--', label='N(0, 1)')
    axs[0].format(title=vc1_name, xlabel='$u\'_L$ (m/s)/$\sigma_{u\'_L}$')
    axs[1].format(title=vc2_name, xlabel='$u\'_T$ (m/s)/$\sigma_{u\'_T}$')
    axs[0].legend(loc='ul', ncols=1)
    axs[1].legend(loc='ul', ncols=1)
    axs.format(yscale='log', ylim=(1e-3, 1), xlim=(-7, 7),
               yformatter='log', suptitle='', fontsize=12, abc=True)
    if savename is not None:
        fig.save(savename, dpi=300)



fig, axs = pplt.subplots(ncols=2, width="5in", sharex=False)
normal_dist = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-1/2 * x**2)
for ax, symb, var in zip(axs, ['d', '+'], ['U_along', 'U_fluctuating']):
    for tau, c in zip(['31D', '15D', '5D'], ['tab:green', 'tab:orange', 'tab:blue']):
        u = results_anomalies[tau][var]
        ustd = results_anomalies[tau][var].std()
        print(tau, var, np.round(ustd*100, 3), 'cm/s')   
        pdf, x_bins = np.histogram(u/ustd, bins=np.linspace(-7, 7, 151), density=True)
        x_center = 1/2*(x_bins[1:] + x_bins[:-1])
        if 'along' in var:
            label = tau
            label_pdf = 'N(0, 1)'
        else:
            label = ''
            label_pdf = ''
        ax.scatter(x_center,
                   pdf, marker=symb,
                   color=c,  label=label, ms=10) 
     
    ax.plot(x_center, normal_dist(x_center), marker='', lw=0.5, color='k', ls='--', label=label_pdf)
    
axs[0].format(title='Longitudinal', xlabel='$u\'_L$ (m/s)/$\sigma_{u\'_L}$')
axs[1].format(title='Transverse', xlabel='$u\'_T$ (m/s)/$\sigma_{u\'_T}$')
# axs[0].legend(loc='ul', ncols=1)
fig.legend(loc='b', ncols=4, ms=20)
axs.format(yscale='log', ylim=(1e-3, 1), xlim=(-7, 7),
           yformatter='log', suptitle='', fontsize=12, abc=True)
fig.save('../figures/fig03_velocity_distribution_tau.pdf', dpi=300)


df = results_anomalies['5D'].copy()
edge_bins = np.arange(0, 800, 25)
df['edge_bin'] = np.digitize(df.edge_dist, bins=edge_bins)
df_stdev = pd.concat({'sigma_ut': df[['edge_bin', 'U_fluctuating']].groupby('edge_bin').std(),
                      'sigma_ul': df[['edge_bin', 'U_along']].groupby('edge_bin').std(),
              'n': df[['edge_bin', 'U_fluctuating']].groupby('edge_bin').count(),
             'd': df[['edge_bin', 'edge_dist']].groupby('edge_bin').mean()}, axis=1)
df_stdev.columns = pd.Index(['sigma_ut', 'sigma_ul', 'n', 'd'])

length_bins = np.arange(0, 50, 2)
df['length_scale'] = df['floe_size']**0.5
df['length_bin'] = np.digitize(df.length_scale, bins=length_bins)
df_stdev_fsd = pd.concat({'sigma_ut': df[['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df[['length_bin', 'U_along']].groupby('length_bin').std(),
              'n': df[['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
             'L': df[['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_stdev_fsd.columns = pd.Index(['sigma_ut', 'sigma_ul', 'n', 'L'])

length_bins = np.arange(0, 50, 2)
df['length_scale'] = df['floe_size']**0.5
df['length_bin'] = np.digitize(df.length_scale, bins=length_bins)
df_stdev_fsd = pd.concat({'sigma_ut': df[['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df[['length_bin', 'U_along']].groupby('length_bin').std(),
              'n': df[['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
             'L': df[['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_stdev_fsd.columns = pd.Index(['sigma_ut', 'sigma_ul', 'n', 'L'])

R = sim_std_dev['u_along (m/s)'].index
L = np.sqrt(np.pi)*R


fig, axs = pplt.subplots(ncols=2, nrows=2, share=False)

normal_dist = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-1/2 * x**2)
for ax, symb, var in zip([axs[0], axs[1]], ['d', '+'], ['U_along', 'U_fluctuating']):
    u = df[var]
    ustd = df[var].std()
    print(tau, var, np.round(ustd*100, 3), 'cm/s')   
    pdf, x_bins = np.histogram(u/ustd, bins=np.linspace(-7, 7, 151), density=True)
    x_center = 1/2*(x_bins[1:] + x_bins[:-1])
    if 'along' in var:
        label_pdf = 'N(0, 1)'
    else:
        label_pdf = ''

    ax.scatter(x_center, pdf, marker=symb, label='', zorder=5) 
    ax.plot(x_center, normal_dist(x_center), marker='',
            lw=0.5, color='k', ls='--', label=label_pdf, zorder=1)
    
# The normalization in np.histogram is not the same as the normalization in
# Minki's matlab routine, to bring them to the same convention, we need to divide by
# the bin spacing. 
dx = np.diff(sim_uacross.index)[0]
   
for col in sim_ualong.columns:
    axs[0].scatter(sim_ualong.index, sim_ualong.loc[:, col]/dx, color='gray',
                m='d', alpha=0.25, label='', zorder=0)
    axs[1].scatter(sim_uacross.index, sim_uacross.loc[:, col]/dx, color='gray',
                alpha=0.25, m='+',  label='', zorder=0)
    
axs[0].format(title='', xlabel='$u\'_L$ (m/s)/$\sigma_{u\'_L}$',
              yscale='log', ylim=(1e-3, 1), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')
axs[1].format(title='', xlabel='$u\'_T$ (m/s)/$\sigma_{u\'_T}$',
              yscale='log', ylim=(1e-3, 1), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')

idx = df_stdev_fsd.n > 300
ax = axs[2]
ax.scatter(df_stdev_fsd.loc[idx, 'L'].values,
        df_stdev_fsd.loc[idx, 'sigma_ul'].values,
        marker='d', label='', color='tab:blue')
ax.scatter(df_stdev_fsd.loc[idx, 'L'].values,
        df_stdev_fsd.loc[idx, 'sigma_ut'].values,
        marker='+', label='', color='tab:blue')
ax.scatter(L, sim_std_dev['u_along (m/s)'],
           marker='d', label='', color='gray')
ax.scatter(L, sim_std_dev['u_across (m/s)'],
           marker='+', label='', color='gray')



idx = df_stdev.n > 300
axs[3].scatter(df_stdev.loc[idx, 'd'].values,
               df_stdev.loc[idx, 'sigma_ul'].values, marker='d', color='tab:blue')
axs[3].scatter(df_stdev.loc[idx, 'd'].values,
               df_stdev.loc[idx, 'sigma_ut'].values, marker='+', color='tab:blue')
axs[3].format(ylim=(0, 0.12), ytickminor=False, xtickminor=False,
         ylabel='$\sigma_{u\'}$', xlabel='Distance to ice edge (km)', fontsize=12)

l = ['Observed', 'Simulated', 'Longitudinal', 'Transverse']
h = [ax.plot([],[], marker=m, color=c, lw=0) for m, c in zip(['o', 'o', 'd', '+'],
                                                       ['tab:blue', 'gray', 'k', 'k'])]
axs[2].format(ylim=(0, 0.12))
axs[3].format(ylim=(0, 0.12))
ax.format(xlabel='Length scale (km)', ylabel='$\sigma_{u\'}$', title='')
ax.legend(h, l, loc='ur', ncols=1)
fig.format(abc=True, fontsize=12, leftlabels=['Velocity distribution', 'Standard Deviation'])
fig.save('../figures/fig04_velocity_obs_sim.pdf', dpi=300)