import os
import pandas as pd
import ultraplot as pplt
import numpy as np
import powerlaw

#### Load dataframe ####
floe_lib_clean = {}
for file in os.listdir('../data/floe_tracker/ift_floe_property_tables/clean/'):
    if 'csv' in file: 
        year = int(file.replace('.csv', '').split('_')[-1])
        floe_lib_clean[year] = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/clean/' + file,
                                         index_col=0, dtype={'classification': str})
        floe_lib_clean[year]['datetime'] = pd.to_datetime(floe_lib_clean[year]['datetime'])

df = pd.concat(floe_lib_clean).reset_index()
df = df.loc[(df.datetime.dt.dayofyear >= 91) & (df.datetime.dt.dayofyear <= 258)]
df['perim_km'] = df.perimeter*.25
df['area_km'] = df.area*.25*.25
df['area_adj_km'] = (np.sqrt(df.area) + 8)**2*.25*.25 # Pixel shift minimizes error against manual
df['doy'] = df.datetime.dt.dayofyear
df['year'] = df.datetime.dt.year

#### Show tracked floes vs all floes ####
fig, ax = pplt.subplots(width=4)
xmin = 41
xmax= 90e3
for year, group in df.groupby(df.datetime.dt.year):
    data = group.area_adj_km
    fit = powerlaw.Fit(data,xmin=xmin, xmax=xmax)
    fit.plot_pdf(color='k', linewidth=0.7, ax=ax, label='', alpha=0.3)
    fit.plot_ccdf(color='k', linewidth=0.7, ax=ax, alpha=0.3)
data = df.area_adj_km
fit = powerlaw.Fit(data,xmin=xmin, xmax=xmax)
fit.plot_pdf(color='k', linewidth=2, ax=ax, label='All floes')
fit.plot_ccdf(color='k', linewidth=2, ax=ax)

for year, group in df.loc[(df.floe_id != 'unmatched')].groupby(
                df.loc[(df.floe_id != 'unmatched')].datetime.dt.year):
    data = group.area_adj_km
    fit = powerlaw.Fit(data,xmin=xmin, xmax=xmax)
    fit.plot_pdf(color='r', linewidth=0.7, ax=ax, label='', alpha=0.3, ls='--')
    fit.plot_ccdf(color='r', linewidth=0.7, ax=ax, alpha=0.3, ls='--')

data = df.loc[(df.floe_id != 'unmatched')].area_adj_km
fit = powerlaw.Fit(data,xmin=xmin, xmax=xmax)
fit.plot_pdf(color='r', linewidth=2, ax=ax, label='Tracked only', ls='--')
fit.plot_ccdf(color='r', linewidth=2, ax=ax, ls='--')

ax.text(1020, 0.15, 'CCDF')
ax.text(1020, 0.00015, 'PDF')

ax.legend(loc='ll', ncols=1)
ax.format(xlabel='Floe area (km$^2$)', ylabel='Probability')
fig.save('../figures/fig13_all_floes_v_tracked_FSD.png')
fig.save('../figures/fig13_all_floes_v_tracked_FSD.pdf')