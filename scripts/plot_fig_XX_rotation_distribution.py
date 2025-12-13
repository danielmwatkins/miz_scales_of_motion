import ultraplot as pplt
import numpy as np
import pandas as pd

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
subset = df_filtered.loc[(df_filtered.qc_flag == 0) & df_filtered.zeta.notnull()].copy()

subset['month'] = subset.datetime.dt.month
subset['year'] = subset.datetime.dt.year
subset['length_scale_km'] = np.sqrt(subset['area_km2'])

pack_subset = subset.loc[subset.nsidc_sic >= 0.85]
miz_subset = subset.loc[subset.nsidc_sic < 0.85]

pack_counts = pack_subset.groupby(['month', 'l_bin']).count()['zeta'].reset_index().pivot_table(
    index='month', columns='l_bin', values='zeta')
pack_summary = pack_subset.loc[:, ['l_bin', 'zeta']].groupby('l_bin').quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95])
pack_summary.reset_index(inplace=True)
pack_summary.columns=['l_bin', 'quantile', 'zeta']
pack_summary = pack_summary.pivot_table(index='l_bin', columns='quantile', values='zeta')

x = pack_subset[['l_bin', 'length_scale_km']].groupby('l_bin').mean()
pack_summary.index = x.loc[pack_summary.index].values.squeeze()

miz_summary = miz_subset.loc[:, ['l_bin', 'zeta']].groupby('l_bin').quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95])
miz_summary.reset_index(inplace=True)
miz_summary.columns=['l_bin', 'quantile', 'zeta']
miz_summary = miz_summary.pivot_table(index='l_bin', columns='quantile', values='zeta')

x = miz_subset[['l_bin', 'length_scale_km']].groupby('l_bin').mean()
miz_summary.index = x.loc[miz_summary.index].values.squeeze()

fig, axs = pplt.subplots(ncols=2, nrows=2, share=False)
for data, ax in zip([pack_summary, miz_summary], axs[0,:]):
    ax.plot(data[0.05], ls='--', lw=1, color='tab:blue')
    ax.plot(data[0.95], ls='--', lw=1, color='tab:blue')
    
    ax.plot(data[0.5], shadedata=[data[0.25], data[0.75]],
            fadedata=[data[0.1], data[0.9]])
    ax.format(ylabel='Rotation rate (rad/day)', ylim=(-1, 1),
              xlocator=np.arange(10, 31, 10), xlim=(6.5, 30),
              xlabel='Floe length scale (km)')
    ax.axhline(0, color='k', ls='--', lw=1)
for data, ax in zip([pack_subset, miz_subset], axs[1,:]):
    counts = data.groupby(['month', 'l_bin']).count()['zeta'].reset_index().pivot_table(
    index='month', columns='l_bin', values='zeta')

    for bin_num in range(2, 9):
        L = data.loc[data.l_bin==bin_num, 'L'].mean()
        v = data.loc[data.l_bin==bin_num, ['month', 'zeta']].groupby('month').var()
        n = counts[bin_num]
        if sum(n > 100) > 1:
            ax.plot(v.where(n > 100), label=str(np.round(L, 0)) + ' km', marker='.')

    ax.legend(ncols=1, loc='ur')
    ax.format(ylabel='Var($\\Omega$)', xlabel='Month', ylim=(0, 0.3))

h = []
for alpha, ls, m in zip([1, 1, 0.25,  0.5], ['-', '--', '', ''], ['', '', 's', 's']):
    h.append(ax.plot([],[],color='tab:blue', alpha=alpha, ls=ls, m=m))
axs[0,0].legend(h, ['Median',  '5-95%', '10-90%', '25-75%'], ncols=1, loc='lr')
axs[0,1].legend(h, ['Median',  '5-95%', '10-90%', '25-75%'], ncols=1, loc='lr')

axs[0,0].format(title='Pack Ice')
axs[1,0].format(title='Pack Ice')
axs[0,1].format(title='MIZ')
axs[1,1].format(title='MIZ')
axs.format(abc=True)

for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/figXX_rotation_rate_distribution.{im}'.format(im=imtype), dpi=300)