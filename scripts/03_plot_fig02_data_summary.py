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

ift_raw = {}
ift_clean = {}
man_images = {}
dataframes = {}
tc_images = {}
fc_images = {}

df = pd.read_csv('../data/floe_tracker/ift_floe_properties.csv', index_col=0)
df['datetime'] = pd.to_datetime(df['datetime'].values)
df_area_val = pd.read_csv('../data/floe_tracker/ift_floe_properties_area_validation.csv', parse_dates=['datetime'])

for date in ['2013-04-24 12:39:09', '2014-05-01 12:14:19']:
    date = pd.to_datetime(date)
    ift_raw[date] = skimage.io.imread(
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

# move earlier if I need to show two dates
fc_images[date] = rio.open(
        '../data/validation_images/{d}/{d}.aqua.falsecolor.250m.tiff'.format(
            d=date.strftime('%Y%m%d'))
        ).read()


imdate = pd.to_datetime('2014-05-01 12:14:19')
overlap_floes = floe_lib_clean.loc[(floe_lib_clean.datetime.dt.year == 2014) & \
    (floe_lib_clean.floe_id != 'unmatched')].groupby('floe_id').filter(
        lambda x: imdate in x.datetime.values)

ref = rio.open(
    '../data/validation_images/{d}/{d}.aqua.truecolor.250m.tiff'.format(
        d=imdate.strftime('%Y%m%d')))

############ Figure 2: Shape detection and area adjustment ################
fig, axs = pplt.subplots(ncols=3, nrows=2, share=False, sharex=False, spany=False)

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

# Overlay raw floes
ax.imshow(np.ma.masked_array(reshape_as_image(ift_raw[date]),
                                reshape_as_image(ift_raw[date])==0),
          color='sky blue', alpha=0.75, extent=[left, right, bottom, top])

# Overlay clean floes
ax.imshow(np.ma.masked_array(reshape_as_image(ift_clean[date]),
                                reshape_as_image(ift_clean[date])==0),
          color='tangerine', alpha=1, extent=[left, right, bottom, top])

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
    ax.imshow(np.ma.masked_array(ift_raw[date], ift_raw[date]==0), c='sky blue', extent=[left, right, bottom, top], alpha=0.75)
    ax.imshow(np.ma.masked_array(ift_clean[date], ift_clean[date]==0), c='tangerine', extent=[left, right, bottom, top], alpha=1)
    ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines == 0), color='b')
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

for imtype in ['png', 'pdf']:
    fig.save('../figures/fig02_algorithm_example.{im}'.format(im=imtype), dpi=300)

