import itertools
import numpy as np
import pandas as pd
import ultraplot as pplt
import os
import pyproj
import sys
import warnings
import xarray as xr
# sys.path.append('/Users/dwatkin2/Documents/research/packages/buoy_processing/')
sys.path.append('../scripts/')
from scipy.interpolate import interp2d
from drifter import compute_velocity
from drifter import compute_along_across_components
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

#### Specify locations for data.
# External datasets not included in archive are stored in a lower 
# sic_loc = '../../../data/nsidc_daily_cdr/'
# motion_loc = '../../../data/nsidc_daily_icemotion/'
ift_loc = '../data/floe_tracker/ift_floe_trajectories.csv'
df_ift = pd.read_csv(ift_loc, index_col=0)
df_ift['datetime'] = pd.to_datetime(df_ift['datetime'].values)
df_ift['area_adj_km2'] = (np.sqrt(df_ift.area) + 8)**2*.25*.25 # 8 pixel shift and convert to km2

# edge_bins = np.arange(0, 800, 25)
# df_ift['edge_bin'] = np.digitize(df_ift.edge_dist_km, bins=edge_bins)


df_ift['L'] = df_ift['area_adj_km2']**0.5
sic = df_ift['nsidc_sic']
df_ift.loc[sic > 1, 'nsidc_sic'] = np.nan

# Additional filter
# I think I added this into the prep script
speed = np.sqrt(df_ift.loc[:, 'u']**2 + df_ift.loc[:, 'v']**2)
mean_u = df_ift.loc[:, 'u'].mean()
mean_v = df_ift.loc[:, 'v'].mean()

z = np.sqrt((df_ift.u - mean_u)**2 + (df_ift.v - mean_v)**2)/np.std(speed)
df_ift['qc_flag'] = 0
df_ift.loc[np.abs(z) > 6, 'qc_flag'] = 1
df_filtered = df_ift.loc[df_ift.qc_flag==0]

df_filtered['l_bin'] = np.digitize(df_filtered['L'], bins=np.arange(0, 60, 5))
df_filtered['l_center'] = [pd.Series(np.arange(2.5, 63, 5), index=np.arange(1, 14))[x] for x in df_filtered['l_bin']]
subset = df_filtered.loc[(df_filtered.qc_flag == 0) & df_filtered.zeta.notnull()]

df_cyc = subset.where(subset.zeta > 0).dropna().pivot_table(
    columns='l_center', values='zeta', index='floe_id')
df_anticyc = -subset.where(subset.zeta < 0).dropna().pivot_table(
    columns='l_center', values='zeta', index='floe_id')

idx_cyc = df_cyc.notnull().sum(axis=0) > 100
idx_anticyc = df_anticyc.notnull().sum(axis=0) > 100

df_cyc = df_cyc.loc[:, idx_cyc].copy()
df_anticyc = df_anticyc.loc[:, idx_anticyc].copy()

sim_rot = pd.read_csv('../data/simulation/rotation_rates.csv')
sim_rot.columns = [c.replace('km', '') for c in sim_rot.columns]
sr_melted = sim_rot.melt()
sr_melted['variable'] = sr_melted['variable'].astype(int)
sr_melted.columns = ['length_scale', 'rotation_rate']
sr_melted['L'] = np.sqrt(np.pi*(sr_melted['length_scale'])**2)

fig, axs = pplt.subplots(ncols=2, sharex=False)
R = sim_rot.where(sim_rot > 0).quantile(0.99, axis=0).interpolate().index.astype(int)
L = np.sqrt(np.pi)*R

for q, lw, m in zip([0.99, 0.95, 0.75, 0.5],
                        [0.5, 1, 2, 3],
                   ['', '', '', '']):
    

    axs[0].plot(L, sim_rot.where(sim_rot > 0).quantile(q, axis=0), #.interpolate(),
                color='k', lw=lw, ls='--', marker=m, zorder=1)
    axs[0].area(L, sim_rot.where(sim_rot > 0).quantile(q, axis=0), #.interpolate(),
               alpha=0.1, color='k', zorder=0)
    
    axs[0].plot(df_cyc.columns, df_cyc.quantile(q, axis=0), #.interpolate(),
                color='tab:blue', lw=lw, marker='.', ls='-', ms=4, zorder=2)

    
    axs[1].plot(L, (-sim_rot.where(sim_rot < 0)).quantile(q, axis=0),# .interpolate(),
                color='k', lw=lw, ls='--', marker=m, zorder=1)
    axs[1].area(L, (-sim_rot.where(sim_rot < 0)).quantile(q, axis=0),#.interpolate(),
               alpha=0.1, color='k', zorder=0)

    axs[1].plot(df_anticyc.columns, df_anticyc.quantile(q, axis=0), #.interpolate(),
                color='tab:blue', lw=lw, ls='-', m='.', ms=4, zorder=2)

axs[0].format(xlocator=np.arange(10, 51, 10),
          xlabel='Floe length scale (km)',
           ylabel='Rotation rate (rad/day)', xlim=(2.5, 47.5))
axs[1].format(xlocator=np.arange(10, 51, 10),
          xlabel='Floe length scale (km)',
           ylabel='Rotation rate (rad/day)', xlim=(2.5, 47.5))


axs[0].format(title='Cyclonic')
axs[1].format(title='Anticyclonic')
#     suptitle='Floe rotation rate distribution by length scale',)

# Legend
l = ['Observation', 'Simulation',
     '99%', '95%', '75%', '50%']
h = [axs[0].plot([],[],c=c, lw=lw, m='', ls=ls)
     for c, lw, ls in zip(['tab:blue', 'gray', 'k', 'k', 'k', 'k'],
                         [2, 2, 0.5, 1, 2, 3], ['-', '--', '-', '-', '-', '-'])]
axs[0].legend(h, l, ncols=1)
fig.format(fontsize=12, abc=True)
fig.save('../figures/fig13_rotation_rate_distribution.pdf', dpi=300)
fig.save('../figures/fig13_rotation_rate_distribution.png', dpi=300)