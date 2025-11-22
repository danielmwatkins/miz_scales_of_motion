"""
Produces the following figures:
1. fig02_algorithm_example.pdf
2. fig03_tracked_floes.pdf
3. fig04_data_availability.pdf
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

ift_raw = {}
ift_clean = {}
man_images = {}
dataframes = {}
tc_images = {}
fc_images = {}
proc_images = {}

df = pd.read_csv('../data/floe_tracker/ift_floe_properties.csv', index_col=0)
df['datetime'] = pd.to_datetime(df['datetime'].values)
df_area_val = pd.read_csv('../data/floe_tracker/ift_floe_properties_area_validation.csv', parse_dates=['datetime'])
plot_dates = ['2013-04-24 12:39:09', '2014-05-01 12:14:19'] 
for date in plot_dates:
    date = pd.to_datetime(date)
    ift_raw[date] = skimage.io.imread(
        '../data/modis_images/{d}/{d}.aqua.labeled_raw.250m.tiff'.format(
            d=date.strftime('%Y%m%d')))
    ift_clean[date] = skimage.io.imread(
        '../data/modis_images/{d}/{d}.aqua.labeled_clean.250m.tiff'.format(
            d=date.strftime('%Y%m%d')))
    tc_images[date] = rio.open(
        '../data/modis_images/{d}/{d}.aqua.truecolor.250m.tiff'.format(
            d=date.strftime('%Y%m%d'))
        ).read()
    fc_images[date] = rio.open(
        '../data/modis_images/{d}/{d}.aqua.falsecolor.250m.tiff'.format(
            d=date.strftime('%Y%m%d'))
        ).read()    
    proc_images[date] = rio.open(
        '../data/modis_images/{d}/{d}.aqua.preprocessed.250m.tiff'.format(
            d=date.strftime('%Y%m%d'))
        ).read()    
    man_images[date] = skimage.io.imread(
        '../data/validation_images/{d}/{d}.aqua.labeled_manual.png'.format(
            d=date.strftime('%Y%m%d')))[:,:,0]


# imdate = pd.to_datetime('2014-05-01 12:14:19')
# overlap_floes = floe_lib_clean.loc[(floe_lib_clean.datetime.dt.year == 2014) & \
#     (floe_lib_clean.floe_id != 'unmatched')].groupby('floe_id').filter(
#         lambda x: imdate in x.datetime.values)


############ Figure 2: Shape detection and area adjustment ################
fig, axs = pplt.subplots(ncols=3, nrows=3, share=False, sharex=False, spany=False)

ref = rio.open(
    '../data/modis_images/{d}/{d}.aqua.truecolor.250m.tiff'.format(
        d=pd.to_datetime(plot_dates[0]).strftime('%Y%m%d')))

left, bottom, right, top = ref.bounds
left /= 1e6
bottom /= 1e6
right /= 1e6
top /= 1e6

inset_left = 0.75
inset_right = 0.95
inset_bottom = -1.85
inset_top =  -1.65

label = ['g.', 'h.']

for row, imdate in zip([0, 1], tc_images):
    for ax, image in zip([axs[row, 0], axs[row, 1]], [tc_images[imdate], fc_images[imdate]]):
        ax.imshow(reshape_as_image(image), extent=[left, right, bottom, top])
        ax.format(ylim=(-2, -1.5), xlim=(0.6, 1.1)) # compare with fig 3?
    ax = axs[row, 2]
    ax.imshow(reshape_as_image(proc_images[imdate]), cmap='mono_r', vmin=0, vmax=255, extent=[left, right, bottom, top])

    # Overlay raw floes
    ax.imshow(np.ma.masked_array(ift_raw[imdate], ift_raw[imdate]==0),
              color='sky blue', alpha=0.75, extent=[left, right, bottom, top])

    # Overlay clean floes
    ax.imshow(np.ma.masked_array(ift_clean[imdate], ift_clean[imdate]==0),
          color='tangerine', alpha=1, extent=[left, right, bottom, top])

    h = [ax.plot([],[], color=c, alpha=a, lw=0, marker='s') 
         for c, a in zip(['sky blue', 'tangerine', 'k'], [0.5, 1, 1])]
    ax.legend(h, ['Non-floes', 'Floes', 'Masked'], loc='ll', ncols=1, alpha=1)  


    ax.plot([inset_left, inset_left, inset_right, inset_right, inset_left],
            [inset_bottom, inset_top, inset_top, inset_bottom, inset_bottom], lw=1, color='k')
    ax.plot([inset_left, inset_left, inset_right, inset_right, inset_left],
            [inset_bottom, inset_top, inset_top, inset_bottom, inset_bottom], lw=1, color='w', ls='--')

    ax.text(inset_left + 0.01, inset_top - 0.03, label[row], color='k', bbox={'facecolor': 'w'})
    
    datestring = imdate.strftime("%Y-%m-%d")
    for col, title in zip([0, 1, 2], ['True Color', 'False Color', 'Processed']):
        axs[row, col].format(title=datestring + " " + title, ylabel='Y ($\\times 10^6$ m)', xlabel='X ($\\times 10^6$ m)')

    ax.format(ylim=(-2, -1.5), xlim=(0.6, 1.1)) # compare with fig 3?
    
axs.format(abc=True)
    
for ax, date in zip(axs[2,:], man_images):
    ax.imshow(reshape_as_image(tc_images[date]), extent=[left, right, bottom, top])
    outlines = man_images[date] - skimage.morphology.erosion(man_images[date], skimage.morphology.disk(4))
    ax.imshow(np.ma.masked_array(ift_raw[date], ift_raw[date]==0), c='sky blue', extent=[left, right, bottom, top], alpha=0.75)
    ax.imshow(np.ma.masked_array(ift_clean[date], ift_clean[date]==0), c='tangerine', extent=[left, right, bottom, top], alpha=1)
    ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines == 0), color='b')
    ax.format(xlim=(inset_left, inset_right), ylim=(inset_bottom, inset_top), xtickminor=False, ytickminor=False, xlocator=0.1, ylocator=0.1,
              title=date.strftime('%Y-%m-%d'), yreverse=False)

h = [ax.plot([],[], ls=ls, m=m, lw=lw, color=c) for lw, c, ls, m in zip([1, 3, 3], ['b', 'tangerine', 'sky blue'], ['-', '', ''], ['', 's', 's'])]
axs[2, 0].legend(h, ['Manual', 'IFT floes', 'IFT non-floes'], loc='ul', ncols=1, alpha=1)
for ax in axs[2, 0:2]:
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
ax = axs[2, 2]
ax.scatter(A_man[idx_all], A_ift[idx_all], marker='.', color='light gray')
ax.scatter(A_man[idx_all], A_adj[idx_all], marker='+', color='light gray')
ax.scatter(A_man[idx], A_ift[idx], marker='.', color='gold')
ax.scatter(A_man[idx], A_adj[idx], marker='+', color='b')
ax.format(ylim=(10, max(A_man.dropna())), xlim=(10, max(A_man.dropna())),
                 yscale='log', xscale='log', xlabel='Manual Area (km$^2$)', ylabel='IFT Area (km$^2$)', title='Area Adjustment')
ax.plot([0, max(A_man.dropna())], [0, max(A_man.dropna())], color='k', ls='--')

h = [ax.plot([],[],marker=m, color=c, lw=lw, ls=ls, alpha=a)
             for a, m, c, lw, ls in zip([0.5, 1, 1], ['.', '+', ''], ['gold', 'b', 'k'], [0,0,1], ['', '', '--'])]
ax.legend(h, ['Initial', 'Adjusted', '1:1'], ncols=1, loc='ul')

for imtype in ['png', 'pdf']:
    fig.save('../figures/fig02_algorithm_example.{im}'.format(im=imtype), dpi=300)

