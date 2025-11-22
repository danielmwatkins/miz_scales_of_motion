"""
Produces the following figures:
1. fig02_algorithm_example.pdf
2. fig03_tracked_floes.pdf
3. fig04_data_availability.pdf
"""

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
floe_lib_clean = pd.concat(floe_lib_clean).reset_index()

trajectories = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
trajectories['datetime'] = pd.to_datetime(trajectories['datetime'])

ift_images = {}
ift_clean = {}
man_images = {}
dataframes = {}
tc_images = {}

df = pd.read_csv('../data/floe_tracker/ift_floe_properties.csv', index_col=0)
df['datetime'] = pd.to_datetime(df['datetime'].values)

df_area_val = pd.read_csv('../data/floe_tracker/ift_floe_properties_area_validation.csv', parse_dates=['datetime'])

for date in ['2013-04-24 12:39:09', '2014-05-01 12:14:19']:
    date = pd.to_datetime(date)
    ift_images[date] = skimage.io.imread(
        '../data/validation_images/{d}/{d}.aqua.labeled_raw.250m.tiff'.format(
            d=date.strftime('%Y%m%d')))
    ift_clean[date] = skimage.io.imread(
        '../data/validation_images/{d}/{d}.aqua.labeled_clean.250m.tiff'.format(
            d=date.strftime('%Y%m%d')))
    tc_images[date] = rio.open(
        '../data/validation_images/{d}/{d}.aqua.truecolor.250m.tiff'.format(
            d=date.strftime('%Y%m%d'))
        ).read()

    man_images[date] = skimage.io.imread(
        '../data/validation_images/{d}/{d}.aqua.labeled_manual.png'.format(
            d=date.strftime('%Y%m%d')))[:,:,0]

############ Figure 2: Shape detection and area adjustment ################
fig, axs = pplt.subplots(ncols=3, nrows=2, share=False, sharex=False, spany=False)

imdate = pd.to_datetime('2014-05-01 12:14:19')
overlap_floes = floe_lib_clean.loc[(floe_lib_clean.datetime.dt.year == 2014) & \
    (floe_lib_clean.floe_id != 'unmatched')].groupby('floe_id').filter(
        lambda x: imdate in x.datetime.values)

ref = rio.open(
    '../data/validation_images/{d}/{d}.aqua.truecolor.250m.tiff'.format(
        d=imdate.strftime('%Y%m%d')))

fc_images = {imdate: rio.open(
        '../data/validation_images/{d}/{d}.aqua.falsecolor.250m.tiff'.format(
            d=imdate.strftime('%Y%m%d'))
        ).read()}

lb_raw_image = rio.open(
    '../data/validation_images/{d}/{d}.aqua.labeled_raw.250m.tiff'.format(
        d=imdate.strftime('%Y%m%d'))).read()
lb_clean_image = rio.open(
    '../data/validation_images/{d}/{d}.aqua.labeled_clean.250m.tiff'.format(
    d=imdate.strftime('%Y%m%d'))).read()

pp_img = rio.open(
    '../data/validation_images/{d}/{d}.aqua.preprocessed.250m.tiff'.format(
    d=imdate.strftime('%Y%m%d'))).read().squeeze()

left, bottom, right, top = ref.bounds
left /= 1e6
bottom /= 1e6
right /= 1e6
top /= 1e6

for ax, image in zip([axs[0, 0], axs[0, 1]], [tc_images[imdate], fc_images[imdate]]):
    ax.imshow(reshape_as_image(image), extent=[left, right, bottom, top])
    
ax = axs[0, 2]
ax.imshow(pp_img, cmap='mono_r', extent=[left, right, bottom, top])

ax.imshow(np.ma.masked_array(reshape_as_image(lb_raw_image),
                                reshape_as_image(lb_raw_image)==0), color='sky blue', alpha=0.75, extent=[left, right, bottom, top])
ax.imshow(np.ma.masked_array(reshape_as_image(lb_clean_image),
                                reshape_as_image(lb_clean_image)==0), color='tangerine', alpha=1, extent=[left, right, bottom, top])

h = [ax.plot([],[], color=c, alpha=a, lw=0, marker='s') for c, a in zip(['sky blue', 'tangerine', 'k'], [0.5, 1, 1])]

ax.legend(h, ['Non-floes', 'Floes', 'Masked'], loc='ll', ncols=1, alpha=1)  
for ax in axs:
    ax.format(ylim=(-2, -1.5), xlim=(0.6, 1.1)) # compare with fig 3
axs.format(abc=True)
for ax, title in zip(axs[0,:], ['True Color', 'False Color', 'Processed']):
    ax.format(title=title, ylabel='Y ($\\times 10^6$ m)', xlabel='X ($\\times 10^6$ m)')

for ax, date in zip(axs[1,:], man_images):
    ax.imshow(reshape_as_image(tc_images[date]), extent=[left, right, bottom, top])
    outlines = man_images[date] - skimage.morphology.erosion(man_images[date], skimage.morphology.disk(4))
    # ax.imshow(np.ma.masked_array(man_images[date], man_images[date]==0), c='k')
    ax.imshow(np.ma.masked_array(ift_images[date], ift_images[date]==0), c='sky blue', extent=[left, right, bottom, top], alpha=0.75)
    ax.imshow(np.ma.masked_array(ift_clean[date], ift_clean[date]==0), c='tangerine', extent=[left, right, bottom, top], alpha=1)
    ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines == 0), color='b')
    # ax.format(xlim=(0.75, 0.95), ylim=(-1.6, -1.4), xtickminor=False, ytickminor=False, xlocator=0.1, ylocator=0.1,
    #           title=date.strftime('%Y-%m-%d'), yreverse=False)
    ax.format(xlim=(0.75, 0.95), ylim=(-1.8, -1.6), xtickminor=False, ytickminor=False, xlocator=0.1, ylocator=0.1,
              title=date.strftime('%Y-%m-%d'), yreverse=False)

h = [ax.plot([],[], ls=ls, m=m, lw=lw, color=c) for lw, c, ls, m in zip([1, 3, 3], ['b', 'tangerine', 'sky blue'], ['-', '', ''], ['', 's', 's'])]
axs[1, 0].legend(h, ['Manual', 'IFT floes', 'IFT non-floes'], loc='ul', ncols=1, alpha=1)

for ax in axs[1, 0:2]:
    ax.format(ylabel='Y ($\\times 10^6$ m)', xlabel='X ($\\times 10^6 m)$')

idx = df_area_val.match_type == 'Good'
A_ift = df_area_val['ift_area']
A_man = df_area_val['manual_area']
A_adj = (A_ift**0.5 + 8)**2

# convert to km2
A_man *= 0.25**2
A_adj *= 0.25**2
A_ift *= 0.25**2
idx_all = df_area_val.final_classification
axs[1, 2].scatter(A_man[idx_all], A_ift[idx_all], marker='.', color='light gray')
axs[1, 2].scatter(A_man[idx_all], A_adj[idx_all], marker='+', color='light gray')
axs[1, 2].scatter(A_man[idx], A_ift[idx], marker='.', color='gold')
axs[1, 2].scatter(A_man[idx], A_adj[idx], marker='+', color='b')
axs[1, 2].format(ylim=(10, max(A_man.dropna())), xlim=(10, max(A_man.dropna())),
                 yscale='log', xscale='log', xlabel='Manual Area (km$^2$)', ylabel='IFT Area (km$^2$)', title='Area Adjustment')
axs[1, 2].plot([0, max(A_man.dropna())], [0, max(A_man.dropna())], color='k', ls='--')

h = [axs[1, 2].plot([],[],marker=m, color=c, lw=lw, ls=ls, alpha=a)
             for a, m, c, lw, ls in zip([0.5, 1, 1], ['.', '+', ''], ['gold', 'b', 'k'], [0,0,1], ['', '', '--'])]
axs[1, 2].legend(h, ['Initial', 'Adjusted', '1:1'], ncols=1, loc='ul')

fig.save('../figures/fig02_algorithm_example.pdf', dpi=300)
fig.save('../figures/fig02_algorithm_example.png', dpi=300)

######## Figure 3: Trajectory example ########

# Dates with multiple tracked floes, all from the Aqua satellite
plot_dates = [pd.to_datetime(x) for x in ['2014-04-27 12:38:45', '2014-04-28 11:43:40',
                                          '2014-04-29 12:26:32', '2014-04-30 11:31:29']]

# load images
tc_images = {}
floe_images = {}

for date in plot_dates:
    tc_images[date] = rio.open('../data/example_images/{d}.aqua.truecolor.250m.tiff'.format(d=date.strftime('%Y%m%d')))
    floe_images[date] = rio.open('../data/example_images/{d}.aqua.labeled_clean.250m.tiff'.format(d=date.strftime('%Y%m%d')))

imdate = pd.to_datetime('2014-04-27 12:38:45')
plot_floes = floe_lib_clean.loc[(floe_lib_clean.floe_id != 'unmatched') & (floe_lib_clean.datetime.dt.year == imdate.year)].groupby('floe_id').filter(lambda x: imdate in x.datetime.values)

# get nearby floes that have enough dates
floe = '2014_01741'
comp_floe_loc = plot_floes.loc[plot_floes.floe_id == floe, ['x_stere', 'y_stere']].mean()
dx = 1e5
dy = 1e5
nearby_floes = plot_floes.loc[np.sqrt((plot_floes.x_stere - comp_floe_loc.x_stere)**2 + \
                                      (plot_floes.y_stere - comp_floe_loc.y_stere)**2) < dx]
nearby_floes = nearby_floes.groupby('floe_id').filter(lambda x: np.sum([d in x.datetime.values for d in plot_dates])>2)
floes = np.unique(nearby_floes.floe_id)

left, bottom, right, top = tc_images[imdate].bounds
left /= 1e3
bottom /= 1e3
right /= 1e3
top /= 1e3

x0 = 850
y0 = -1700
left -= x0
right -= x0
top -= y0
bottom -= y0

fig, axs = pplt.subplots(ncols=2, nrows=2, width=6, spanx=False, spany=False)

colors = [c['color'] for c in pplt.Cycle('Dark2', 12)]

for ax, date in zip(axs,plot_dates):
    ax.imshow(reshape_as_image(tc_images[date].read()), extent=[left, right, bottom, top])
    image = floe_images[date].read().squeeze()
    
    outlines = image - skimage.morphology.erosion(image, skimage.morphology.disk(4))

    for c, floe in zip(colors, floes):
        df_floe = plot_floes.loc[plot_floes.floe_id == floe].set_index('datetime')

        # Plot the main tracked floe
        if date in df_floe.index:
            ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines != df_floe.loc[date, 'label']), color=c)

            ax.plot(df_floe.loc[df_floe.index <= date, 'x_stere'].values/1e3 - x0,
                    df_floe.loc[df_floe.index <= date, 'y_stere'].values/1e3 - y0, color=c, marker='.', facecolor='w')
            ax.plot(df_floe.loc[date, 'x_stere']/1e3 - x0,
                    df_floe.loc[date, 'y_stere']/1e3 - y0, color=c, marker='.')
            
    ax.format(ylim=(-100, 100), xlim=(-100, 100), title=date, ylabel='Y (km)', xlabel='X (km)')
axs.format(abc=True)
fig.save('../figures/fig03_tracked_floes.png', dpi=300)
fig.save('../figures/fig03_tracked_floes.pdf', dpi=300)
pplt.close(fig)

######## Figure 4: Summary of data availability
import cartopy.crs as ccrs
import warnings
warnings.simplefilter('ignore')
pplt.rc['cartopy.circular'] = False
pplt.rc['reso'] = 'med'

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

fig.save('../figures/fig04_data_availability.pdf')
fig.save('../figures/fig04_data_availability.png')
