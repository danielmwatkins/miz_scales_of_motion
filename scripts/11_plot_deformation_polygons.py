import rasterio as rio
from rasterio.plot import reshape_as_image
import skimage
import pandas as pd
import numpy as np
import proplot as pplt

# Need interpolated and exact dates
date_interp = pd.to_datetime('2013-04-24 12:00:00')
date_exact = pd.to_datetime('2013-04-24 13:55:54')

# Full dataframe to get the floe labels
df_full_res = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/clean/ift_clean_floe_properties_{y}.csv'.format(y=date_exact.year), index_col=0)
df_full_res['datetime'] = pd.to_datetime(df_full_res['datetime'])
df_full_date = df_full_res.loc[df_full_res.datetime == date_exact]


# Load deformation data
df = pd.read_csv('../data/deformation/sampled_results.csv', index_col=0)
df['datetime'] = pd.to_datetime(df['datetime'])
df_date = df.loc[df.datetime == date_interp]
satellite = 'terra'
tc_image = rio.open('../data/modis_imagery/{d}.{s}.truecolor.250m.tiff'.format(d=date_exact.strftime('%Y%m%d'), s=satellite))
lb_clean_image = rio.open('../data/modis_imagery/{d}.{s}.labeled_clean.250m.tiff'.format(d=date_exact.strftime('%Y%m%d'), s=satellite)).read()

left, bottom, right, top = tc_image.bounds
left /= 1e6
bottom /= 1e6
right /= 1e6
top /= 1e6

ymin = -1.7e6
ymax = -1.45e6
xmin = 0.76e6
xmax = xmin + (ymax - ymin)

fig, axs = pplt.subplots(ncols=3, width=8, aspect=1, spanx=False)

for ax, log_bin, color in zip(axs, [2, 4, 6], ['b', 'b', 'b']):
    ax.imshow(reshape_as_image(tc_image.read()), extent=[left, right, bottom, top])
    ax.imshow(np.ma.masked_array(reshape_as_image(lb_clean_image),
                             mask=reshape_as_image(lb_clean_image)==0), extent=[left, right, bottom, top], color='tangerine', alpha=0.9)

    # could add step to get the label and color the floes at vertices!
    
    plot_sel = df_date.loc[(df_date.log_bin == log_bin) & \
                            (df_date.x1.between(xmin, xmax) & (df_date.y1.between(ymin, ymax)))].set_index('triangle_number')
    plot_sel = plot_sel.loc[plot_sel.no_overlap_sample]
    plot_df = pd.DataFrame(columns=['floe1', 'floe2', 'floe3', 'x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'label1', 'label2', 'label3'], index=plot_sel.index)
    for triangle_number in plot_df.index:
        for i in range(1, 4):
            col = 'floe' + str(i)
            floe_id = plot_sel.loc[triangle_number, col]
            plot_df.loc[triangle_number, col] = floe_id
            plot_df.loc[triangle_number, 'x' + str(i)] = float(df_full_date.loc[df_full_date.floe_id == floe_id, 'x_stere'].values)
            plot_df.loc[triangle_number, 'y' + str(i)] = float(df_full_date.loc[df_full_date.floe_id == floe_id, 'y_stere'].values)
            plot_df.loc[triangle_number, 'label' + str(i)] = int(df_full_date.loc[df_full_date.floe_id == floe_id, 'label'].iloc[0])
    # Potentially add step to outline the floes at the triangle vertices
    filtered_image = lb_clean_image.copy()
    labels = np.unique(list(plot_df['label1'].values) + list(plot_df['label2']) + list(plot_df['label3']))
    labels = np.unique(list(plot_df['label1'].values) + list(plot_df['label2']) + list(plot_df['label3']))
    mask = np.zeros(filtered_image.shape)
    for l in labels:
        mask += filtered_image == l
    mask = (mask > 0).astype(int)

    # Outlines -- could also fill in the full floe if wanted
    outlines = reshape_as_image(mask)[:,:,0] - skimage.morphology.erosion(reshape_as_image(mask)[:,:,0], skimage.morphology.disk(4))
    ax.pcolorfast(np.linspace(left, right, outlines.shape[1]),
                  np.linspace(top, bottom, outlines.shape[0]),
                  np.ma.masked_array(outlines, mask=outlines == 0), color='k')

    # Plot the triangles
    for tri in plot_df.index:
        ax.plot(plot_df.loc[tri, ['x1', 'x2', 'x3', 'x1']].astype(float).values/1e6,
                plot_df.loc[tri, ['y1', 'y2', 'y3', 'y1']].astype(float).values/1e6, color=color, lw=1)

    lscale = plot_sel['L'].mean()
    ax.text(xmax/1e6 - 0.09, ymax/1e6 - 0.02, text = 'L $\\approx$ ' + str(np.round(lscale, 1)) + ' km', color='b')
    ax.format(ylabel='Y ($\\times 10^6$ m)', xlabel='X ($\\times 10^6$ m)')
    ax.format(ylim=(ymin/1e6, ymax/1e6), xlim=(xmin/1e6, xmax/1e6))
axs.format(abc=True)
fig.save('../figures/fig06_polygon_example.png', dpi=300)
fig.save('../figures/fig06_polygon_example.pdf', dpi=300)