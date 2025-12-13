import ultraplot as uplt
import pandas as pd
import numpy as np

dr = pd.date_range('2020-04-01', '2020-12-01', freq='1MS')

nsidc_sie = pd.read_csv('../data/cdr_sea_ice_extent.csv', index_col=0, parse_dates = True)
# asi_sie = pd.read_csv('../data/asi_sea_ice_extent.csv', index_col=0, parse_dates = True)
# asi_sie = asi_sie.where(asi_sie['sea_ice_extent'] > 1)

## Sea ice extent 
fig, axs = uplt.subplots(nrows=2, share=False)
ax = axs[0]
for year, group in nsidc_sie.groupby(nsidc_sie.index.year):
    ax.plot(group.index.dayofyear,
            group['sea_ice_extent'].values/1e9, alpha=0.25, color='tab:blue')
mean_extent = nsidc_sie.groupby(nsidc_sie.index.dayofyear).mean()
ax.plot(mean_extent.index,
            mean_extent['sea_ice_extent'].values/1e9, alpha=1, lw=3, color='tab:blue')
ax.format(ylabel='Sea ice extent (1000 km$^2$)',  xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', xrotation=45, fontsize=12, xlim=(85, 265), title='Sea ice extent')

## MIZ fraction
ax = axs[1]
for year, group in nsidc_sie.groupby(nsidc_sie.index.year):
    ax.plot(group.index.dayofyear,
            group['miz_ice_extent'].values/group['sea_ice_extent'].values * 100,
           alpha=0.25, color='tab:blue')
nsidc_sie['miz_percent'] = nsidc_sie['miz_ice_extent'] / nsidc_sie['sea_ice_extent']

mean_percent = nsidc_sie.groupby(nsidc_sie.index.dayofyear).mean()
ax.plot(mean_percent.index,
            mean_percent['miz_percent'].values*100, alpha=1, lw=3, color='tab:blue')
ax.format(ylabel='MIZ Fraction (%)', xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', xrotation=45, fontsize=12, xlim=(85, 265), title='MIZ fraction')

h = []
for alpha, ls, m in zip([1, 0.5, 0.25], ['-', '', ''], ['', 's', 's']):
    h.append(ax.plot([],[],color='tab:blue', alpha=alpha, ls=ls, m=m))
h = [ax.plot([], [], color='tab:blue', lw=1, alpha=1),
     ax.plot([], [], color='tab:blue', lw=3, alpha=1)]
axs[0].legend(h, ['Individual Years', '2003-2020 Mean'], ncols=1,  loc='ur')
axs[1].legend(h, ['Individual Years', '2003-2020 Mean'], ncols=1, loc='ul')

axs.format(abc=True)
for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/fig09_cdr_sea_ice_extent.{im}'.format(im=imtype), dpi=300)