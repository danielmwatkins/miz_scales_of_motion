import proplot as pplt
import pandas as pd
import numpy as np
import os

floe_lib_raw = {}
for file in os.listdir('../data/floe_tracker/ift_floe_property_tables/raw/'):
    if 'csv' in file:
        year = int(file.replace('.csv', '').split('_')[-1])
        floe_lib_raw[year] = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/raw/' + file,
                                         index_col=0, dtype={'classification': str})
        floe_lib_raw[year]['datetime'] = pd.to_datetime(floe_lib_raw[year]['datetime'])
floe_lib_raw = pd.concat(floe_lib_raw)
floe_lib_clean = pd.read_csv('../data/floe_tracker/ift_floe_properties.csv', index_col=0)
floe_lib_clean['datetime'] = pd.to_datetime(floe_lib_clean['datetime'])

trajectories = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
trajectories['datetime'] = pd.to_datetime(trajectories['datetime'])

import cartopy.crs as ccrs
import warnings
warnings.simplefilter('ignore')
pplt.rc['cartopy.circular'] = False
pplt.rc['reso'] = 'med'

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
fig, axs = pplt.subplots(width=6, proj='npstere', proj_kw={'lon_0': -45}, ncols=3)
axs.format(land=True, coast=True, landzorder=0, landcolor='k', facecolor='w')

for ax in axs:
    ax.set_extent([0.2e6, 1.3e6, -2.15e6, -0.5e6], crs=crs)  
    ax.plot([0.3e6, 0.3e6 + 200e3], [-2.05e6, -2.05e6], lw=4, color='w', zorder=3, transform=crs)
    ax.text(0.3e6, -2.05e6, '200 km', color='w')
# Left: All floes
ax = axs[0]
data, _, _ = np.histogram2d(floe_lib_clean.x_stere, floe_lib_clean.y_stere, bins=[xbins, ybins])
data = pd.DataFrame(data, index=xc, columns=yc)
c = ax.pcolormesh(xc, yc, data.where(data >= 30).T, transform=crs, cmap='spectral_r', vmin=0, vmax=2000, N=20, extend='max')
ax.colorbar(c, label='Count', loc='b')

# Middle: Tracked floes (note: this is all floe images that have been linked, not unique floes
ax = axs[1]
data, _, _ = np.histogram2d(floe_lib_clean.loc[floe_lib_clean.floe_id != 'unmatched', 'x_stere'],
                            floe_lib_clean.loc[floe_lib_clean.floe_id != 'unmatched', 'y_stere'],
                            bins=[xbins, ybins])
data = pd.DataFrame(data, index=xc, columns=yc)
c = ax.pcolormesh(xc, yc, data.where(data >= 30).T, transform=crs, cmap='spectral_r', vmin=0, vmax=2000, N=20, extend='max')
ax.colorbar(c, label='Count', loc='b')

# Right: Trajectories
ax = axs[2]
for floe_id, track in trajectories.groupby('floe_id'):
    if len(track) > 7:
        ax.plot(track.x_stere.values, track.y_stere.values, lw=1, transform=crs)
axs[0].format(title='All floes')
axs[1].format(title='Tracked floes')
axs[2].format(title='Long trajectories')
axs.format(abc=True)
fig.save('../figures/fig01_spatial_histogram.pdf', dpi=300)

import rasterio as rio
from rasterio.plot import reshape_as_image
imdate = pd.to_datetime('2014-05-01 12:14:19')
overlap_floes = floe_lib_clean.loc[(floe_lib_clean.datetime.dt.year == 2014) & \
    (floe_lib_clean.floe_id != 'unmatched')].groupby('floe_id').filter(
        lambda x: imdate in x.datetime.values)

fig, axs = pplt.subplots(ncols=3, sharey=True, nrows=1, width=8, sharex=False, spany=False)
for row, date in zip([0, 1], ['20140501']):
    tc_image = rio.open('../data/modis_imagery/{d}.aqua.truecolor.250m.tiff'.format(d=date))
    fc_image = rio.open('../data/modis_imagery/{d}.aqua.falsecolor.250m.tiff'.format(d=date))
    pp_image = rio.open('../data/modis_imagery/{d}.aqua.preprocessed.250m.tiff'.format(d=date)).read().squeeze()
    
    left, bottom, right, top = tc_image.bounds
    left /= 1e6
    bottom /= 1e6
    right /= 1e6
    top /= 1e6
    
    for ax, image in zip([axs[0], axs[1]], [tc_image, fc_image]):
        ax.imshow(reshape_as_image(image.read()), extent=[left, right, bottom, top])
        
    
    ax = axs[2]
    ax.imshow(pp_image, cmap='mono_r', extent=[left, right, bottom, top])
    
    lb_raw_image = rio.open('../data/modis_imagery/{d}.aqua.labeled_raw.250m.tiff'.format(d=date)).read()
    lb_clean_image = rio.open('../data/modis_imagery/{d}.aqua.labeled_clean.250m.tiff'.format(d=date)).read()
    ax.imshow(np.ma.masked_array(reshape_as_image(lb_raw_image),
                                    reshape_as_image(lb_raw_image)==0), color='r', alpha=0.5, extent=[left, right, bottom, top])
    ax.imshow(np.ma.masked_array(reshape_as_image(lb_clean_image),
                                    reshape_as_image(lb_clean_image)==0), color='b', alpha=1, extent=[left, right, bottom, top])

    h = [ax.plot([],[], color=c, alpha=a, lw=0, marker='s') for c, a in zip(['r', 'b', 'k'], [0.5, 1, 1])]

    ax.legend(h, ['Non-floes', 'Floes', 'Masked'], loc='ll', ncols=1)  

axs.format(abc=True)
for ax, title in zip(axs, ['True Color', 'False Color', 'Processed']):
    ax.format(title=title, ylabel='Y (m $\\times 10^6$)', xlabel='X (m $\\times 10^6)$')
fig.save('../figures/fig02_image_processing_1x3.pdf', dpi=300)

data = []
for year in range(2003, 2021):
    n = len(floe_lib_raw.loc[floe_lib_raw.datetime.dt.year == year])
    amax = floe_lib_raw.loc[floe_lib_raw.datetime.dt.year == year].area.max()
    amin = floe_lib_raw.loc[floe_lib_raw.datetime.dt.year == year].area.min()
    n_passing = len(floe_lib_raw.loc[(floe_lib_raw.datetime.dt.year == year) & floe_lib_raw.final_classification])
    # n_tracked = len(np.unique(trajectories.loc[trajectories.datetime.dt.year == year, 'floe_id'])) - 1 # remove "untracked" from the list
    n_tracked = len(trajectories.loc[(trajectories.datetime.dt.year == year) & (trajectories.floe_id != 'unmatched'), 'x_stere'].dropna())
    n_rotation = len(trajectories.loc[trajectories.datetime.dt.year == year, 'zeta'].dropna())
    data.append([year, amax, amin, n, n_passing, n_tracked, n_rotation])
data_table = pd.DataFrame(data, columns=['year', 'max_pixels', 'min_pixels', 'n_init', 'n_passing', 'n_tracked', 'n_rotation'])
data_table.set_index('year', inplace=True)

fig, axs= pplt.subplots(nrows=2, share=False)
ax = axs[0]
ax.plot(data_table.index.values, data_table['n_init'].values, label='Raw', marker='.')
ax.plot(data_table.index.values, data_table['n_passing'].values, label='Clean', marker='+')
ax.plot(data_table.index.values, data_table['n_tracked'].values, label='Tracked', marker='^')
ax.plot(data_table.index.values, data_table['n_rotation'].values, label='Rotation', marker='s')
ax.legend(loc='ll', ncols=2)
ax.format(ylabel='Count', xlabel='Year', ylim=(100, 125000), yscale='log', title='Data availability')

ax = axs[1]
c = pplt.Cycle('spectral', 18)
colors = {year: c['color'] for c, year in zip(c, np.arange(2003, 2021))}
h = []
for year, group in trajectories.groupby(trajectories.datetime.dt.year):
    n = group.groupby('floe_id').apply(lambda x: len(x), include_groups=False)
    x, bins = np.histogram(n, bins=np.arange(0.5, 20))
    xc = 0.5*(bins[1:] + bins[:-1])
    h.append(ax.plot(xc[1:], x[1:], marker='.', color=colors[year]))
ax.legend(h, [x for x in colors], loc='ur', ncols=2, order='F')
ax.format(ylabel='Count', xlabel='Length (days)', title='Length of trajectories', xlim=(2, 12))
axs.format(abc=True)
fig.save('../figures/fig03_data_availability.pdf', dpi=300)