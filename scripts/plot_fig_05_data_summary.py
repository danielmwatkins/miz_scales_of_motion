"""
Produces the following figure:
fig05_data_availability.pdf
"""

import cartopy.crs as ccrs
import numpy as np
import os
import ultraplot as pplt
import pandas as pd
import rasterio as rio
from rasterio.plot import reshape_as_image
import skimage
import warnings

warnings.simplefilter('ignore')
pplt.rc['cartopy.circular'] = False
pplt.rc['reso'] = 'med'

# Data ingest: shape properties and trajectories
floe_lib_raw = {}
for file in os.listdir('../data/floe_tracker/ift_floe_property_tables/raw/'):
    if 'csv' in file: 
        year = int(file.replace('.csv', '').split('_')[-1])
        floe_lib_raw[year] = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/raw/' + file,
                                         index_col=0, dtype={'classification': str})
        floe_lib_raw[year]['datetime'] = pd.to_datetime(floe_lib_raw[year]['datetime'])
floe_lib_raw = pd.concat(floe_lib_raw)

floe_lib_clean = {}
for file in os.listdir('../data/floe_tracker/ift_floe_property_tables/clean/'):
    if 'csv' in file: 
        year = int(file.replace('.csv', '').split('_')[-1])
        floe_lib_clean[year] = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/clean/' + file,
                                         index_col=0, dtype={'classification': str})
        floe_lib_clean[year]['datetime'] = pd.to_datetime(floe_lib_clean[year]['datetime'])
floe_lib_clean = pd.concat(floe_lib_clean).reset_index()

trajectories = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
trajectories['datetime'] = pd.to_datetime(trajectories['datetime'])


data = []
for year in range(2003, 2021):
    n = len(floe_lib_raw.loc[floe_lib_raw.datetime.dt.year == year])
    amax = floe_lib_raw.loc[floe_lib_raw.datetime.dt.year == year].area.max()
    amin = floe_lib_raw.loc[floe_lib_raw.datetime.dt.year == year].area.min()
    n_passing = len(floe_lib_raw.loc[(floe_lib_raw.datetime.dt.year == year) & floe_lib_raw.final_classification])
    # n_tracked = len(np.unique(trajectories.loc[trajectories.datetime.dt.year == year, 'floe_id'])) - 1 # remove "untracked" from the list
    n_vel = len(trajectories.loc[trajectories.datetime.dt.year == year, 'u'].dropna())
    n_tracked = len(np.unique(trajectories.loc[(trajectories.datetime.dt.year == year), 'floe_id'].dropna()))
    n_rotation = len(trajectories.loc[trajectories.datetime.dt.year == year, 'zeta'].dropna())
    data.append([year, amax, amin, n, n_passing, n_vel, n_tracked, n_rotation])
    
data_table = pd.DataFrame(data, columns=['year', 'max_pixels', 'min_pixels', 'n_init','n_passing', 'n_vel',  'n_tracked', 'n_rotation'])
data_table.set_index('year', inplace=True)

floe_lib_clean['doy'] = floe_lib_clean.datetime.dt.dayofyear
floe_lib_clean['year'] = floe_lib_clean.datetime.dt.year
count_range = floe_lib_clean.groupby(['year', 'doy']).count().pivot_table(index='year', columns='doy', values='datetime').quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=0)
smoothed_count_range = count_range.T.rolling(15, center=True).mean()

year = 2004
xmin = 0.2e6
xmax = 1.2e6
ymin = -2.5e6
ymax = -0.25e6
dx = 25e3
xbins = np.arange(xmin, xmax, dx)
ybins = np.arange(ymin, ymax, dx)
xc = 0.5*(xbins[1:] + xbins[:-1])
yc = 0.5*(ybins[1:] + ybins[:-1])

crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
fig, axs = pplt.subplots(width=8, height=6.5, proj={1: 'npstere', 2: 'npstere', 3: 'npstere', 4: None, 5: None, 6: None},
                         proj_kw={'lon_0': -45}, ncols=3, nrows=2, hratios=[2, 1], share=False)
axs.format(land=True, coast=True, landzorder=0, landcolor='k', facecolor='w')
for ax in axs[0:3]:
    ax.set_extent([247326, 1115678, -635759, -2089839], crs=crs)  
    ax.plot([0.3e6, 0.3e6 + 200e3], [-2.05e6, -2.05e6], lw=4, color='w', zorder=3, transform=crs)
    ax.text(0.3e6, -2.05e6, '200 km', color='w')

# Left: All floes
ax = axs[0, 0]
data, _, _ = np.histogram2d(floe_lib_clean.x_stere, floe_lib_clean.y_stere, bins=[xbins, ybins])
data = pd.DataFrame(data, index=xc, columns=yc)
c = ax.pcolormesh(xc, yc, data.where(data >= 30).T, transform=crs, cmap='spectral_r', vmin=0, vmax=2000, N=20, extend='max')
ax.colorbar(c, label='Count', loc='b')

# Middle: Tracked floes (note: this is all floe images that have been linked, not unique floes
ax = axs[0, 1]
data, _, _ = np.histogram2d(floe_lib_clean.loc[floe_lib_clean.floe_id != 'unmatched', 'x_stere'],
                            floe_lib_clean.loc[floe_lib_clean.floe_id != 'unmatched', 'y_stere'],
                            bins=[xbins, ybins])
data = pd.DataFrame(data, index=xc, columns=yc)
c = ax.pcolormesh(xc, yc, data.where(data >= 30).T, transform=crs, cmap='spectral_r', vmin=0, vmax=2000, N=20, extend='max')
ax.colorbar(c, label='Count', loc='b')

# Right: Trajectories
ax = axs[0, 2]
for floe_id, track in trajectories.groupby('floe_id'):
    if len(track) > 7:
        ax.plot(track.x_stere.values, track.y_stere.values, lw=1, transform=crs)
axs[0, 0].format(title='All floes')
axs[0, 1].format(title='Tracked floes')
axs[0, 2].format(title='Long trajectories')

# Bottom left: Counts
ax = axs[1, 0]
ax.plot(data_table.index.values, data_table['n_init'].values, label='Raw', marker='.')
ax.plot(data_table.index.values, data_table['n_passing'].values, label='Clean', marker='+')
ax.plot(data_table.index.values, data_table['n_tracked'].values, label='Tracked', marker='^')
ax.plot(data_table.index.values, data_table['n_rotation'].values, label='Rotation', marker='s')
ax.legend(loc='lr', ncols=2)
ax.format(ylabel='Count', xlabel='Year', ylim=(10, 150000), yscale='log',
          title='Number of floes by year', ylocator=(1e2, 1e3, 1e4, 1e5), yformatter=['$10^2$', '$10^3$', '$10^4$', '$10^5$'])

# Bottom middle: Floes per image

ax = axs[1, 1]
ax.plot(smoothed_count_range[0.5], color='tab:blue',
        shadedata=[smoothed_count_range[0.25],
                   smoothed_count_range[0.75]],
       fadedata=[smoothed_count_range[0.1],
                 smoothed_count_range[0.9]])
dr = pd.date_range('2020-04-01', '2020-09-01', freq='1MS')
ax.format(xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr], xrotation=45)
h = []
for alpha, ls, m in zip([1, 0.5, 0.25], ['-', '', ''], ['', 's', 's']):
    h.append(ax.plot([],[],color='tab:blue', alpha=alpha, ls=ls, m=m))
ax.legend(h, ['Median', '25-75%', '1-90%'], ncols=1, loc='ur')
ax.format(ylabel='Count', xlabel='', title='Number of floes per image')
axs.format(abc=True)

# Bottom right: Trajectory lengths
ax = axs[1, 2]
c = pplt.Cycle('spectral', 18)
colors = {year: c['color'] for c, year in zip(c, np.arange(2003, 2021))}
h = []
for year, group in trajectories.groupby(trajectories.datetime.dt.year):
    n = group.groupby('floe_id').apply(lambda x: len(x), include_groups=False)
    x, bins = np.histogram(n, bins=np.arange(0.5, 20))
    xc = 0.5*(bins[1:] + bins[:-1])
    idx = x > 0
    x = x[idx]
    xc = xc[idx]
    
    h.append(ax.plot(xc[1:], x[1:], marker='.', color=colors[year]))
ax.format(title='Trajectory length distribution', ylabel='Count', xlabel='Length (days)', yscale='log',
         ylocator=(1, 1e1, 1e2, 1e3, 1e4), yformatter=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
ax.legend(h[::3], [str(x) for x in range(2003, 2021)][::3], loc='ur', ncols=1, order='F')

fig.save('../figures/fig05_data_availability.pdf')
fig.save('../figures/fig05_data_availability.png')
