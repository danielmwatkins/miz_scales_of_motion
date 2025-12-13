import os
import pandas as pd
import ultraplot as pplt
import numpy as np
import powerlaw


# Load floe data
floe_lib_clean = {}
for file in os.listdir('../data/floe_tracker/ift_floe_property_tables/with_nsidc/'):
    if 'csv' in file: 
        year = int(file.replace('.csv', '').split('_')[-1])
        floe_lib_clean[year] = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/with_nsidc/' + file,
                                         index_col=0, dtype={'classification': str})
        floe_lib_clean[year]['datetime'] = pd.to_datetime(floe_lib_clean[year]['datetime'])

df = pd.concat(floe_lib_clean).reset_index()
df = df.loc[(df.datetime.dt.dayofyear >= 91) & (df.datetime.dt.dayofyear <= 258)]
df['perim_km'] = df.perimeter*.25
df['area_km'] = df.area*.25*.25
df['area_adj_km'] = (np.sqrt(df.area) + 8)**2*.25*.25 # Pixel shift minimizes error against manual
df['doy'] = df.datetime.dt.dayofyear
df['year'] = df.datetime.dt.year


df['band_1_reflectance'] = df['tc_channel0']/255
df['month'] = df.datetime.dt.month
df['n'] = df.groupby('datetime').transform(lambda x: len(x))['area']
df['length_scale_km'] = df['area_adj_km']**0.5

dr = pd.date_range('2020-04-01', '2020-09-01', freq='1MS')



fig, axs = pplt.subplots(ncols=4, share=False, nrows=2)

idx = df.n > 30
idx = idx & df.nsidc_sic.between(0.15, 0.85)
idx = idx & df.final_classification
for col, var, ylims in zip([0, 1, 2, 3],
                          ['length_scale_km', 'circularity', 'band_1_reflectance', 'nsidc_sic'],
                          [(4, 20), (0.2, 1), (0.5, 1), (0, 1)]):
    # shade data
    ax = axs[0,col]
    n = df.loc[idx, [var, 'doy']].groupby('doy').count()
    med = df.loc[idx, [var, 'doy']].groupby('doy').median().where(n > 500)
    p90 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.90).where(n > 500)
    p10 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.10).where(n > 500)
    p75 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.75).where(n > 500)
    p25 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.25).where(n > 500)
    ax.plot(med[var], fadedata=[p90[var], p10[var]],
            shadedata=[p75[var], p25[var]], marker='.', color='slateblue')
    ax.format(ylabel=var.replace('_', ' '),
              xlocator=dr.dayofyear,
              xformatter=[d.strftime('%b') for d in dr],
              xlabel='', fontsize=12, ylim=ylims)

idx = df.n > 30
idx = idx & df.nsidc_sic.between(0.85, 1)
idx = idx & df.final_classification
for col, var, ylims in zip([0, 1, 2, 3],
                          ['length_scale_km', 'circularity', 'band_1_reflectance', 'nsidc_sic'],
                          [(4, 20), (0.2, 1), (0.5, 1), (0, 1)]):
    ax = axs[0,col]
    n = df.loc[idx, [var, 'doy']].groupby('doy').count()
    med = df.loc[idx, [var, 'doy']].groupby('doy').median().where(n > 500)
    p90 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.90).where(n > 500)
    p10 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.10).where(n > 500)
    p75 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.75).where(n > 500)
    p25 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.25).where(n > 500)
    
    # overlay lines
    c = 'tab:green'
    ax.plot(med[var], marker='.', color=c, ms=3, label='MIZ')
    ax.plot(p75[var], ls='-', lw=1, color=c, label='')
    ax.plot(p25[var], ls='-', lw=1, color=c, label='')
    ax.plot(p90[var], ls='--', lw=1, color=c, label='')
    ax.plot(p10[var], ls='--', lw=1, color=c, label='')
    ax.format(ylabel=var.replace('_', ' '),
              xlocator=dr.dayofyear,
              xformatter=[d.strftime('%b') for d in dr],
              xlabel='', fontsize=12, ylim=ylims)
axs[0,0].format(ylabel='Length scale (km)', title='Floe Size')
axs[0,1].format(ylabel='Circularity', title='Floe Shape')
axs[0,2].format(ylabel='Band 1 Reflectance', title='Floe Reflectance')
axs[0,3].format(ylabel='Ice Fraction', title='CDR Sea Ice Fraction')

# axs[0,2].legend(ncols=1, loc='ll')

h = []
for c in ['tab:green', 'slateblue']:
    h.append(ax.plot([],[],color=c, marker='s',  lw=0))
for ax in axs[0,:]:
    ax.legend(h, ['Pack Ice', 'MIZ'], ncols=1, loc='lr')


idx = df.n > 30
idx = idx & df.edge_dist_km.between(0, 100) & (df.coast_dist_km > 100)
idx = idx & df.final_classification
for col, var, ylims in zip([0, 1, 2, 3],
                          ['length_scale_km', 'circularity', 'band_1_reflectance', 'nsidc_sic'],
                          [(4, 20), (0.2, 1), (0.5, 1), (0, 1)]):
    # shade data
    ax = axs[1, col]
    n = df.loc[idx, [var, 'doy']].groupby('doy').count()
    med = df.loc[idx, [var, 'doy']].groupby('doy').median().where(n > 500)
    p90 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.90).where(n > 500)
    p10 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.10).where(n > 500)
    p75 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.75).where(n > 500)
    p25 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.25).where(n > 500)
    ax.plot(med[var], fadedata=[p90[var], p10[var]],
            shadedata=[p75[var], p25[var]], marker='.', color='tab:blue', label='Edge')
    ax.format(ylabel=var.replace('_', ' '),
              xlocator=dr.dayofyear,
              xformatter=[d.strftime('%b') for d in dr],
              xlabel='', fontsize=12, ylim=ylims)

idx = df.n > 30
idx = idx & df.coast_dist_km.between(0, 100) & (df.edge_dist_km > 100)
idx = idx & df.final_classification
for col, var, ylims in zip([0, 1, 2, 3],
                          ['length_scale_km', 'circularity', 'band_1_reflectance', 'nsidc_sic'],
                          [(4, 20), (0.2, 1), (0.5, 1), (0, 1)]):
    ax = axs[1, col]
    n = df.loc[idx, [var, 'doy']].groupby('doy').count()
    med = df.loc[idx, [var, 'doy']].groupby('doy').median().where(n > 500)
    p90 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.90).where(n > 500)
    p10 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.10).where(n > 500)
    p75 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.75).where(n > 500)
    p25 = df.loc[idx, [var, 'doy']].groupby('doy').quantile(0.25).where(n > 500)
    
    # overlay lines
    ax.plot(med[var], marker='.', color='tab:orange', ms=3, label='Coastal')
    ax.plot(p75[var], ls='-', lw=1, color='tab:orange', label='')
    ax.plot(p25[var], ls='-', lw=1, color='tab:orange', label='')
    ax.plot(p90[var], ls='--', lw=1, color='tab:orange', label='')
    ax.plot(p10[var], ls='--', lw=1, color='tab:orange', label='')
    ax.format(ylabel=var.replace('_', ' '),
              xlocator=dr.dayofyear,
              xformatter=[d.strftime('%b') for d in dr],
              xlabel='', fontsize=12, ylim=ylims)
axs[1, 0].format(ylabel='Length scale (km)', title='Floe Size')
axs[1, 1].format(ylabel='Circularity', title='Floe Shape')
axs[1, 2].format(ylabel='Band 1 Reflectance', title='Floe Reflectance')
axs[1, 3].format(ylabel='Ice Fraction', title='CDR Sea Ice Fraction')
# axs[1, 2].legend(ncols=1, loc='ll')
axs.format(abc=True)

h = []
for c in ['tab:blue', 'tab:orange']:
    h.append(ax.plot([],[],color=c, marker='s',  lw=0))
for ax in axs[1,:]:
    ax.legend(h, ['Near Ice Edge', 'Near Coast'], ncols=1, loc='lr')

# Legends
h = []
for alpha, ls, m in zip([1, 0.5, 0.25], ['-', '', ''], ['', 's', 's']):
    h.append(ax.plot([],[],color='k', alpha=alpha, ls=ls, m=m))

for lw, ls, m in zip([2, 1, 1], ['-', '-', '--'], ['.', '', '']):
    h.append(ax.plot([],[],color='k', alpha=1, ls=ls, m=m))

fig.legend(h, ['Pack Ice (SIC > 85%)',
               'MIZ (SIC 15-85%)',
               'Edge Dist < 100 km',
               'Coast Dist < 100 km',
               'Median', '25-75%', '10-90%',
               'Median', '25-75%', '10-90%'], ncols=3, loc='b', order='F')
for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/fig08_floe_properties_seasonality.{im}'.format(im=imtype), dpi=300)