"""
Produces the following figure:
fig04_tracked_floes.pdf
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

######## Figure 3: Trajectory example ########

# Dates with multiple tracked floes, all from the Aqua satellite
plot_dates = [pd.to_datetime(x) for x in ['2014-04-27 12:38:45', '2014-04-28 11:43:40',
                                          '2014-04-29 12:26:32', '2014-04-30 11:31:29']]

# load images
tc_images = {}
clean_images = {}
raw_images = {}


for date in plot_dates:
    tc_images[date] = rio.open('../data/modis_images/fig4/{d}.aqua.truecolor.250m.tiff'.format(d=date.strftime('%Y%m%d')))
    clean_images[date] = skimage.io.imread('../data/modis_images/fig4/{d}.aqua.labeled_clean.250m.tiff'.format(d=date.strftime('%Y%m%d')))
    raw_images[date] = skimage.io.imread('../data/modis_images/fig4/{d}.aqua.labeled_raw.250m.tiff'.format(
            d=date.strftime('%Y%m%d')))

imdate = pd.to_datetime('2014-04-27 12:38:45')
# plot_floes = floe_lib_clean.loc[(floe_lib_clean.floe_id != 'unmatched') & (floe_lib_clean.datetime.dt.year == imdate.year)].groupby('floe_id').filter(lambda x: imdate in x.datetime.values)

plot_floes = floe_lib_clean.loc[(floe_lib_clean.floe_id != 'unmatched') & (floe_lib_clean.datetime.dt.year == imdate.year)].groupby('floe_id').filter(lambda x: len(x.loc[x.satellite == 'aqua']) >= 3)

# get nearby floes that have enough dates
floe = '2014_01741'
comp_floe_loc = plot_floes.loc[plot_floes.floe_id == floe, ['x_stere', 'y_stere']].mean()
dx = 1e5
dy = 1e5
nearby_floes = plot_floes.loc[np.sqrt((plot_floes.x_stere - comp_floe_loc.x_stere)**2 + \
                                      (plot_floes.y_stere - comp_floe_loc.y_stere)**2) < 3*dx]
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

colors = [c['color'] for c in pplt.Cycle('Dark2', len(floes))]

for ax, date in zip(axs,plot_dates):
    ax.imshow(reshape_as_image(tc_images[date].read()), extent=[left, right, bottom, top])
    # Overlay raw floes
    ax.imshow(np.ma.masked_array(raw_images[date], (raw_images[date]==0) | (clean_images[date] > 0)),
              color='sky blue', alpha=0.5, extent=[left, right, bottom, top])

    # Overlay clean floes
    ax.imshow(np.ma.masked_array(clean_images[date], clean_images[date]==0),
          color='tangerine', alpha=0.5, extent=[left, right, bottom, top])

    image = clean_images[date]
    
    outlines = skimage.morphology.dilation(image, skimage.morphology.disk(3)) - image

    for c, floe in zip(colors, floes):
        df_floe = plot_floes.loc[plot_floes.floe_id == floe].set_index('datetime')

        if date in df_floe.index:
            ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines != df_floe.loc[date, 'label']), color=c)

            ax.plot(df_floe.loc[df_floe.index <= date, 'x_stere'].values/1e3 - x0,
                    df_floe.loc[df_floe.index <= date, 'y_stere'].values/1e3 - y0, color=c, marker='.', facecolor='w')
            ax.plot(df_floe.loc[date, 'x_stere']/1e3 - x0,
                    df_floe.loc[date, 'y_stere']/1e3 - y0, color=c, marker='.')
            
    ax.format(ylim=(-100, 100), xlim=(-100, 100), title=date.strftime("%Y-%m-%d %H:%M"), ylabel='Y (km)', xlabel='X (km)')
axs.format(abc=True)

for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/fig04_tracked_floes.{im}'.format(im=imtype), dpi=300)


