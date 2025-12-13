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

# All data, all seasons
df_edge = pd.concat({'sigma_ut': df_comp[['edge_bin', 'U_fluctuating']].groupby('edge_bin').std(),
                      'sigma_ul': df_comp[['edge_bin', 'U_along']].groupby('edge_bin').std(),
                     'mean_ut': np.abs(df_comp[['edge_bin', 
                                                'U_fluctuating']]).groupby('edge_bin').mean(),
                      'mean_ul': np.abs(df_comp[['edge_bin', 'U_along']]).groupby('edge_bin').mean(),
              'n': df_comp[['edge_bin', 'U_fluctuating']].groupby('edge_bin').count(),
             'd': df_comp[['edge_bin', 'edge_dist_km']].groupby('edge_bin').mean()}, axis=1)
df_edge.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul',  'n', 'd'])

df_lscale = pd.concat({'sigma_ut': df_comp[['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df_comp[['length_bin', 'U_along']].groupby('length_bin').std(),
                      'mean_ut': df_comp[['edge_bin', 'U_fluctuating']].groupby('edge_bin').mean(),
                      'mean_ul': df_comp[['edge_bin', 'U_along']].groupby('edge_bin').mean(),
                      'n': df_comp[['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
                      'L': df_comp[['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_lscale.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul', 'n', 'L'])

#### Divide by sea ice concentration
idx = df_comp.nsidc_sic > 0.85
df_pack_edge = pd.concat({'sigma_ut': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').std(),
                      'sigma_ul': df_comp.loc[idx, ['edge_bin', 'U_along']].groupby('edge_bin').std(),
                     'mean_ut': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').mean(),
                      'mean_ul': df_comp.loc[idx, ['edge_bin', 'U_along']].groupby('edge_bin').mean(),
              'n': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').count(),
             'd': df_comp.loc[idx, ['edge_bin', 'edge_dist_km']].groupby('edge_bin').mean()}, axis=1)
df_pack_edge.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul',  'n', 'd'])

df_pack_lscale = pd.concat({'sigma_ut': df_comp.loc[idx, ['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df_comp.loc[idx, ['length_bin', 'U_along']].groupby('length_bin').std(),
                      'mean_ut': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').mean(),
                      'mean_ul': df_comp.loc[idx, ['edge_bin', 'U_along']].groupby('edge_bin').mean(),
                      'n': df_comp.loc[idx, ['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
                      'L': df_comp.loc[idx, ['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_pack_lscale.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul', 'n', 'L'])


idx = df_comp.nsidc_sic.between(0.15, 0.85)
df_miz_edge = pd.concat({'sigma_ut': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').std(),
                      'sigma_ul': df_comp.loc[idx, ['edge_bin', 'U_along']].groupby('edge_bin').std(),
                     'mean_ut': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').mean(),
                      'mean_ul': df_comp.loc[idx, ['edge_bin', 'U_along']].groupby('edge_bin').mean(),
              'n': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').count(),
             'd': df_comp.loc[idx, ['edge_bin', 'edge_dist_km']].groupby('edge_bin').mean()}, axis=1)
df_miz_edge.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul',  'n', 'd'])

df_miz_lscale = pd.concat({'sigma_ut': df_comp.loc[idx, ['length_bin', 'U_fluctuating']].groupby('length_bin').std(),
                      'sigma_ul': df_comp.loc[idx, ['length_bin', 'U_along']].groupby('length_bin').std(),
                      'mean_ut': df_comp.loc[idx, ['edge_bin', 'U_fluctuating']].groupby('edge_bin').mean(),
                      'mean_ul': df_comp.loc[idx, ['edge_bin', 'U_along']].groupby('edge_bin').mean(),
                      'n': df_comp.loc[idx, ['length_bin', 'U_fluctuating']].groupby('length_bin').count(),
                      'L': df_comp.loc[idx, ['length_bin', 'length_scale']].groupby('length_bin').mean()}, axis=1)
df_miz_lscale.columns = pd.Index(['sigma_ut', 'sigma_ul', 'mean_ut', 'mean_ul', 'n', 'L'])


## Directional anomalies
sigma = 1
normal_dist = lambda x: 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2 * (x/sigma)**2)

fig, axs = pplt.subplots(height=6, nrows=2, ncols=3, share=False)
x_bins = np.linspace(-0.6, 0.6, 51)

#### All time scales, comparison with NSIDC
for tau, ls in zip(['5D', '15D', '31D'], ['-', '--', ':']):
    df_temp = df_ift.copy()
    df_temp['u'] = df_temp['u'] - df_temp['u' + tau + '_nsidc']
    df_temp['v'] = df_temp['v'] - df_temp['v' + tau + '_nsidc']
    df_rot_ift = compute_along_across_components(df_temp, uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')

    df_temp = df_ift.copy()
    df_temp['u'] = df_temp['u_nsidc'] - df_temp['u' + tau + '_nsidc']
    df_temp['v'] = df_temp['v_nsidc'] - df_temp['v' + tau + '_nsidc']
    df_rot_nsidc = compute_along_across_components(df_temp, uvar='u', vvar='v',
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
        axs[0, 1].plot(x_center, v_pdf, color=color, label=label, ls=ls)
    comp = df_ift.copy()
    comp['u'] = comp['u'] - comp['u_nsidc']
    comp['v'] = comp['v'] - comp['v_nsidc']
    df_comp = compute_along_across_components(comp, uvar='u', vvar='v',
                                        umean='u' + tau + '_nsidc',
                                        vmean='u' + tau + '_nsidc')
    # u = df_comp['U_along']
    # v = df_comp['U_fluctuating']
    # u_pdf, _ = np.histogram(u, bins=x_bins, density=True)
    # v_pdf, _ = np.histogram(v, bins=x_bins, density=True)
    # x_center = 1/2*(x_bins[1:] + x_bins[:-1])

    label = 'Diff'
    if tau != '5D':
        label = ''
        
    # axs[0, 0].plot(x_center, u_pdf, color='tab:blue', label=label, ls=ls, lw=1)
    # axs[0, 1].plot(x_center, v_pdf, color='tab:blue', label=label, ls=ls, lw=1)
axs[0,0].format(title='', ylabel='Probability')
axs[0,1].format(title='', ylabel='Probability')
axs[0,0].format(xlabel='$u\'_L$ (m/s)', yscale='log', xlim=(-0.4, 0.4), ylim=(0.1, 20), abc=True)
axs[0,1].format(xlabel='$u\'_T$ (m/s)', yscale='log', xlim=(-0.4, 0.4), ylim=(0.1, 20), abc=True)
h = [axs[0,0].plot([],[], lw=1, color=c) for c in ['r', 'k']]
h += [axs[0,0].plot([],[], lw=1, color='gray', ls=ls) for ls in ['-', '--', ':']]
axs[0,0].legend(h, ['IFT', 'NSIDC', '$\\tau=$5D', '$\\tau=$15D', '$\\tau=$31D'], loc='ul', ncols=1)

axs.format(abc=True)

tau = '5D'
comp = df_ift.copy()
comp['u'] = comp['u'] - comp['u5D_nsidc']
comp['v'] = comp['v'] - comp['v5D_nsidc']
df_comp = compute_along_across_components(comp, uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')

#### Fluctuating Velocity Distributions
for ax, symb, var in zip([axs[1,0], axs[1,1]], ['d', 'o'], ['U_along', 'U_fluctuating']):
    for c, idx in zip(['tab:green', 'slateblue'], 
                          [df_comp.nsidc_sic > 0.85, df_comp.nsidc_sic.between(0.15, 0.85)]):
        ustd = df_comp.loc[idx, var].std()
        u = df_comp.loc[idx, var]
        print(tau, var, np.round(ustd*100, 3), 'cm/s')   
        pdf, x_bins = np.histogram(u/ustd, bins=np.linspace(-7, 7, 151), density=True)
        x_center = 1/2*(x_bins[1:] + x_bins[:-1])
        
        label_pdf = 'N(0, 1)'
        
        ax.scatter(x_center, pdf, marker=symb, zorder=5, color=c, label='') 
    
        train = df_comp.loc[idx].dropna(subset=var).sample(1000, replace=False)
        test = df_comp.loc[idx].dropna(subset=var)
        test = test.loc[[x for x in test.index if x not in train.index]]
        abs_u_train = np.abs(train[var])/np.std(train[var])
        abs_u_test = np.abs(test[var])/np.std(test[var])
        exp_loc, exp_scale = stats.expon.fit(abs_u_train, floc=0)
        print('Fitted exponential dist. scale param:', np.round(exp_scale, 2))

        expon_dist = stats.expon(loc=0, scale=exp_scale).pdf
        normal_dist = stats.norm(loc=0, scale=1).pdf

        if c == 'slateblue':
            ax.plot(x_center, normal_dist(x_center), marker='',
                lw=1, color='k', ls='--', label='N(0, 1)', zorder=10)
            ax.plot(x_center, 0.5*expon_dist(np.abs(x_center)), marker='',
            lw=1, color='k', ls=':', label='Exp({s})'.format(s=np.round(exp_scale, 2)), zorder=10)

   
axs[1,0].format(title='', xlabel='$u\'_L / \\sigma_{u\'_L}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')
axs[1,0].legend(ncols=1)
axs[1,1].format(title='', xlabel='$u\'_T / \\sigma_{u\'_T}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')
axs[1,1].legend(ncols=1)
l = ['Pack Ice', 'MIZ']
h = [ax.plot([],[], marker='s', color=c, lw=0, ls='')
     for c in ['tab:green', 'slateblue']]
axs[1,0].legend(h, l, loc='ur', ncols=1)
axs[1,1].legend(h, l, loc='ur', ncols=1)


ax = axs[0,2]
for data, color in zip([df_pack_lscale, df_miz_lscale],
                       ['tab:green', 'slateblue']):
    idx = data.n > 300
    ax.plot(data.loc[idx, 'L'].values,
            data.loc[idx, 'sigma_ul'].values,
            marker='d', label='', color=color)
    ax.plot(data.loc[idx, 'L'].values,
            data.loc[idx, 'sigma_ut'].values, ls='--',
            marker='o', label='', color=color)
    ax.format(xlabel='Length scale (km)', 
              ylabel='$\\sigma_{u\'}$', title='',
              ylim=(0, 0.12), xlim=(5, 30), yscale='linear')
ax = axs[1,2]
for data, color in zip([df_pack_edge, df_miz_edge],
                       [ 'tab:green', 'slateblue']):
    idx = data.n > 300
    ax.plot(data.loc[idx, 'd'].values,
            data.loc[idx, 'sigma_ul'].values,
            marker='d', label='', color=color)
    ax.plot(data.loc[idx, 'd'].values,
            data.loc[idx, 'sigma_ut'].values, ls='--',
            marker='o', label='', color=color)
    ax.format(xlabel='Edge distance (km)', 
              ylabel='$\\sigma_{u\'}$', title='',
              ylim=(0, 0.12), xlim=(0, 400), yscale='linear')

l = ['Longitudinal', 'Transverse', 'Pack Ice', 'MIZ']
h = [ax.plot([],[], marker=m, color=c, lw=lw, ls=ls)
     for m, c, lw, ls in zip(['d', 'o', 's', 's'],
                              ['k', 'k', 'tab:green', 'slateblue'],
                              [1, 1, 0, 0], ['-', '--', '', ''])]
axs[0, 2].legend(h, l, ncols=1, loc='ll')
axs[1, 2].legend(h, l, ncols=1, loc='ll')

fig.format(fontsize=12)
for imtype in ['pdf', 'png']:
    fig.save('../figures/{im}/figXX_simpler_velocity_dist.{im}'.format(im=imtype), dpi=300)

##### Column ordered (option 2) ######
fig, axs = pplt.subplots(height=6, nrows=2, ncols=3, share=False)
x_bins = np.linspace(-0.6, 0.6, 51)

#### All time scales, comparison with NSIDC
for tau, ls in zip(['5D', '15D', '31D'], ['-', '--', ':']):
    df_temp = df_ift.copy()
    df_temp['u'] = df_temp['u'] - df_temp['u' + tau + '_nsidc']
    df_temp['v'] = df_temp['v'] - df_temp['v' + tau + '_nsidc']
    df_rot_ift = compute_along_across_components(df_temp, uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')

    df_temp = df_ift.copy()
    df_temp['u'] = df_temp['u_nsidc'] - df_temp['u' + tau + '_nsidc']
    df_temp['v'] = df_temp['v_nsidc'] - df_temp['v' + tau + '_nsidc']
    df_rot_nsidc = compute_along_across_components(df_temp, uvar='u', vvar='v',
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

    label = 'Diff'
    if tau != '5D':
        label = ''
        
axs[0,0].format(title='', ylabel='Probability')
axs[1,0].format(title='', ylabel='Probability')
axs[0,0].format(xlabel='$u\'_L$ (m/s)', yscale='log', xlim=(-0.4, 0.4), ylim=(0.1, 20), abc=True)
axs[1,0].format(xlabel='$u\'_T$ (m/s)', yscale='log', xlim=(-0.4, 0.4), ylim=(0.1, 20), abc=True)
h = [axs[0,0].plot([],[], lw=1, color=c) for c in ['r', 'k']]
h += [axs[0,0].plot([],[], lw=1, color='gray', ls=ls) for ls in ['-', '--', ':']]
axs[0,0].legend(h, ['IFT', 'NSIDC', '$\\tau=$5D', '$\\tau=$15D', '$\\tau=$31D'], loc='ul', ncols=1)

axs.format(abc=True)

tau = '5D'
comp = df_ift.copy()
comp['u'] = comp['u'] - comp['u5D_nsidc']
comp['v'] = comp['v'] - comp['v5D_nsidc']
df_comp = compute_along_across_components(comp, uvar='u', vvar='v',
                                    umean='u' + tau + '_nsidc',
                                    vmean='u' + tau + '_nsidc')



#### Fluctuating Velocity Distributions
for ax, symb, var in zip([axs[0,1], axs[1,1]], ['d', 'o'], ['U_along', 'U_fluctuating']):
    for c, idx in zip(['tab:green', 'slateblue'], 
                          [df_comp.nsidc_sic > 0.85, df_comp.nsidc_sic.between(0.15, 0.85)]):
        ustd = df_comp.loc[idx, var].std()
        u = df_comp.loc[idx, var]
        print(tau, var, np.round(ustd*100, 3), 'cm/s')   
        pdf, x_bins = np.histogram(u/ustd, bins=np.linspace(-7, 7, 151), density=True)
        x_center = 1/2*(x_bins[1:] + x_bins[:-1])
        
        label_pdf = 'N(0, 1)'
        
        ax.scatter(x_center, pdf, marker=symb, zorder=5, color=c, label='') 
    
        train = df_comp.loc[idx].dropna(subset=var).sample(1000, replace=False)
        test = df_comp.loc[idx].dropna(subset=var)
        test = test.loc[[x for x in test.index if x not in train.index]]
        abs_u_train = np.abs(train[var])/np.std(train[var])
        abs_u_test = np.abs(test[var])/np.std(test[var])
        exp_loc, exp_scale = stats.expon.fit(abs_u_train, floc=0)
        print('Fitted exponential dist. scale param:', np.round(exp_scale, 2))

        expon_dist = stats.expon(loc=0, scale=exp_scale).pdf
        normal_dist = stats.norm(loc=0, scale=1).pdf

        if c == 'slateblue':
            ax.plot(x_center, normal_dist(x_center), marker='',
                lw=1, color='k', ls='--', label='N(0, 1)', zorder=10)
            ax.plot(x_center, 0.5*expon_dist(np.abs(x_center)), marker='',
            lw=1, color='k', ls=':', label='Exp({s})'.format(s=np.round(exp_scale, 2)), zorder=10)

   
axs[0, 1].format(title='', xlabel='$u\'_L / \\sigma_{u\'_L}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')
axs[0, 1].legend(ncols=1)
axs[1,1].format(title='', xlabel='$u\'_T / \\sigma_{u\'_T}$',
              yscale='log', ylim=(1e-4, 1.2), xlim=(-7, 7),
              yformatter='log', ylabel='PDF')
axs[1,1].legend(ncols=1)
l = ['Pack Ice', 'MIZ']
h = [ax.plot([],[], marker='s', color=c, lw=0, ls='')
     for c in ['tab:green', 'slateblue']]
axs[0, 1].legend(h, l, loc='ur', ncols=1)
axs[1,1].legend(h, l, loc='ur', ncols=1)


ax = axs[0,2]
for data, color in zip([df_pack_lscale, df_miz_lscale],
                       ['tab:green', 'slateblue']):
    idx = data.n > 300
    ax.plot(data.loc[idx, 'L'].values,
            data.loc[idx, 'sigma_ul'].values,
            marker='d', label='', color=color)
    ax.plot(data.loc[idx, 'L'].values,
            data.loc[idx, 'sigma_ut'].values, ls='--',
            marker='o', label='', color=color)
    ax.format(xlabel='Length scale (km)', 
              ylabel='$\\sigma_{u\'}$', title='',
              ylim=(0, 0.12), xlim=(5, 30), yscale='linear')
ax = axs[1,2]
for data, color in zip([df_pack_edge, df_miz_edge],
                       [ 'tab:green', 'slateblue']):
    idx = data.n > 300
    ax.plot(data.loc[idx, 'd'].values,
            data.loc[idx, 'sigma_ul'].values,
            marker='d', label='', color=color)
    ax.plot(data.loc[idx, 'd'].values,
            data.loc[idx, 'sigma_ut'].values, ls='--',
            marker='o', label='', color=color)
    ax.format(xlabel='Edge distance (km)', 
              ylabel='$\\sigma_{u\'}$', title='',
              ylim=(0, 0.12), xlim=(0, 400), yscale='linear')

l = ['Longitudinal', 'Transverse', 'Pack Ice', 'MIZ']
h = [ax.plot([],[], marker=m, color=c, lw=lw, ls=ls)
     for m, c, lw, ls in zip(['d', 'o', 's', 's'],
                              ['k', 'k', 'tab:green', 'slateblue'],
                              [1, 1, 0, 0], ['-', '--', '', ''])]
axs[0, 2].legend(h, l, ncols=1, loc='ll')
axs[1, 2].legend(h, l, ncols=1, loc='ll')

fig.format(fontsize=12)
for imtype in ['pdf', 'png']:
    fig.save('../figures/{im}/figXX_simpler_velocity_dist_option2.{im}'.format(im=imtype), dpi=300)