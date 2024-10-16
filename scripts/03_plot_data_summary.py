import cartopy.crs as ccrs
import numpy as np
import os
import proplot as pplt
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
floe_lib_clean = pd.concat(floe_lib_clean)

trajectories = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
trajectories['datetime'] = pd.to_datetime(trajectories['datetime'])


ift_images = {}
ift_clean = {}
man_images = {}
dataframes = {}
tc_images = {}

df = pd.read_csv('../data/floe_tracker/ift_floe_properties.csv', index_col=0)
df['datetime'] = pd.to_datetime(df['datetime'].values)

for date in ['2013-04-24 12:39:09', '2014-05-01 12:14:19']:
    date = pd.to_datetime(date)
    ift_images[date] = skimage.io.imread('../data/modis_imagery/{d}.aqua.labeled_raw.250m.tiff'.format(d=date.strftime('%Y%m%d')))
    ift_clean[date] = skimage.io.imread('../data/modis_imagery/{d}.aqua.labeled_clean.250m.tiff'.format(d=date.strftime('%Y%m%d')))
    tc_images[date] = rio.open('../data/modis_imagery/{d}.aqua.truecolor.250m.tiff'.format(d=date.strftime('%Y%m%d'))).read()

    im_manual = skimage.io.imread('../data/validation_images/{d}.aqua.labeled_manual.png'.format(d=date.strftime('%Y%m%d')))
    man_images[date] = skimage.measure.label(im_manual[:,:,0])
    dataframes[date] = df.loc[df.datetime == date].reset_index(drop=True)

    dataframes[date]['manual_label'] = np.nan
    dataframes[date]['manual_area'] = np.nan
    
    for region in skimage.measure.regionprops(man_images[date]):
        label = np.unique(ift_images[date][(ift_images[date] > 0) & (man_images[date] == region.label)])
        if len(label) == 1:
            dataframes[date].loc[dataframes[date].label == label[0], 'manual_label'] = region.label
            dataframes[date].loc[dataframes[date].label == label[0], 'manual_area'] = region.area
        


############ Figure 1: Shape detection and area adjustment ################
fig, axs = pplt.subplots(ncols=3, nrows=2, share=False, sharex=False, spany=False)

imdate = pd.to_datetime('2014-05-01 12:14:19')
overlap_floes = floe_lib_clean.loc[(floe_lib_clean.datetime.dt.year == 2014) & \
    (floe_lib_clean.floe_id != 'unmatched')].groupby('floe_id').filter(
        lambda x: imdate in x.datetime.values)

date = '20140501'
tc_image = rio.open('../data/modis_imagery/{d}.aqua.truecolor.250m.tiff'.format(d=date))
fc_image = rio.open('../data/modis_imagery/{d}.aqua.falsecolor.250m.tiff'.format(d=date))
pp_image = rio.open('../data/modis_imagery/{d}.aqua.preprocessed.250m.tiff'.format(d=date)).read().squeeze()

left, bottom, right, top = tc_image.bounds
left /= 1e6
bottom /= 1e6
right /= 1e6
top /= 1e6

for ax, image in zip([axs[0, 0], axs[0, 1]], [tc_image, fc_image]):
    ax.imshow(reshape_as_image(image.read()), extent=[left, right, bottom, top])
    
    
ax = axs[0, 2]
ax.imshow(pp_image, cmap='mono_r', extent=[left, right, bottom, top])

lb_raw_image = rio.open('../data/modis_imagery/{d}.aqua.labeled_raw.250m.tiff'.format(d=date)).read()
lb_clean_image = rio.open('../data/modis_imagery/{d}.aqua.labeled_clean.250m.tiff'.format(d=date)).read()
ax.imshow(np.ma.masked_array(reshape_as_image(lb_raw_image),
                                reshape_as_image(lb_raw_image)==0), color='r', alpha=0.75, extent=[left, right, bottom, top])
ax.imshow(np.ma.masked_array(reshape_as_image(lb_clean_image),
                                reshape_as_image(lb_clean_image)==0), color='gold', alpha=1, extent=[left, right, bottom, top])

h = [ax.plot([],[], color=c, alpha=a, lw=0, marker='s') for c, a in zip(['r', 'gold', 'k'], [0.5, 1, 1])]

ax.legend(h, ['Non-floes', 'Floes', 'Masked'], loc='ll', ncols=1, alpha=1)  
for ax in axs:
    ax.format(ylim=(-2, -1.5), xlim=(0.6, 1.1))
axs.format(abc=True)
for ax, title in zip(axs[0,:], ['True Color', 'False Color', 'Processed']):
    ax.format(title=title, ylabel='Y (m $\\times 10^6$)', xlabel='X (m $\\times 10^6)$')



for ax, date in zip(axs[1,:], man_images):
    ax.imshow(reshape_as_image(tc_images[date]), extent=[left, right, bottom, top])
    outlines = man_images[date] - skimage.morphology.erosion(man_images[date], skimage.morphology.disk(4))
    # ax.imshow(np.ma.masked_array(man_images[date], man_images[date]==0), c='k')
    ax.imshow(np.ma.masked_array(ift_images[date], ift_images[date]==0), c='r', extent=[left, right, bottom, top], alpha=0.75)
    ax.imshow(np.ma.masked_array(ift_clean[date], ift_clean[date]==0), c='gold', extent=[left, right, bottom, top], alpha=1)
    ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines == 0), color='b')
    ax.format(xlim=(0.75, 0.95), ylim=(-1.6, -1.4), xtickminor=False, ytickminor=False, xlocator=0.1, ylocator=0.1,
              title=date.strftime('%Y-%m-%d'), yreverse=False)

h = [ax.plot([],[], ls=ls, m=m, lw=lw, color=c) for lw, c, ls, m in zip([1, 3, 3], ['k', 'gold', 'r'], ['-', '', ''], ['', 's', 's'])]
axs[1, 0].legend(h, ['Manual', 'IFT floes', 'IFT non-floes'], loc='ul', ncols=1, alpha=1)

for ax in axs[1, 0:2]:
    ax.format(ylabel='Y (m $\\times 10^6$)', xlabel='X (m $\\times 10^6)$')


for date in dataframes:
    df_date = dataframes[date]
    A_ift = df_date.loc[df_date.manual_area.notnull(), 'area'].values
    A_man = df_date.loc[df_date.manual_area.notnull(), 'manual_area'].values
    A_adj = (A_ift**0.5 + 6)**2
    
    # convert to km2
    A_man *= 0.25**2
    A_adj *= 0.25**2
    A_ift *= 0.25**2
    
    axs[1, 2].scatter(A_man, A_ift, marker='.', color='gold')
    axs[1, 2].scatter(A_man, A_adj, marker='+', color='b')
axs[1, 2].format(ylim=(10, max(A_man)), xlim=(10, max(A_man)),
                 yscale='log', xscale='log', xlabel='Manual', ylabel='Automatic', title='Area Adjustment')
axs[1, 2].plot([0, max(A_man)], [0, max(A_man)], color='k', ls='--')

h = [axs[1, 2].plot([],[],marker=m, color=c, lw=lw, ls=ls, alpha=a)
             for a, m, c, lw, ls in zip([0.5, 1, 1], ['.', '+', ''], ['gold', 'b', 'k'], [0,0,1], ['', '', '--'])]
axs[1, 2].legend(h, ['Raw', 'Adjusted', '1:1'], ncols=1, loc='ul')

fig.save('../figures/fig02_algorithm_example.pdf', dpi=300)
fig.save('../figures/fig02_algorithm_example.png', dpi=300)


######## Figure 2: Trajectory example and rotation rates #########



######## Figure 3: Summary of data availability

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