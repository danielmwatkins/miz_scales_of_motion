"""
Plotting seasonality of the FSD
"""
import os
import pandas as pd
import ultraplot as pplt
import numpy as np
import powerlaw

#### Seasonality of the FSD ####

# Load FSD Data
results = []
for year in range(2003, 2021):
    results.append(pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_{y}.csv'.format(y=year), parse_dates=['date']))
results = pd.concat(results, axis=0)
results_alt = pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_all_years_by_DOY.csv')
dr = pd.date_range('2020-04-01', '2020-09-01', freq='1MS')

doy = results.groupby('month').mean()['doy']

# Load floe data
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

# Load sea ice extent data
df_nsidc = pd.read_csv('../data/nsidc_greenland_sea_ice_extent.csv', parse_dates=True, index_col=0)

fig, axs = pplt.subplots(ncols=4, nrows=1, share=False)

## Panel 1: FSD Seasonality
ax = axs[0]
n_threshold = 1 # check what this one is doing -- is it just to see if the month has anything in it?
month_threshold = 300
variable = 'alpha_tpl'
time_idx = (results.month != 3) & (results.month <= 9)
df_pivot = results.loc[time_idx & (results.n > n_threshold)].pivot_table(index='date', values=variable, columns='month')
month_counts = df_pivot.notnull().sum() 
month_idx = [m for m in month_counts.index if month_counts[m] > month_threshold]
_ = ax.box(doy[month_idx].values/30, df_pivot[month_idx], facecolor='w', alpha=0.7, zorder=1, lw=1.5, marker='')
_ = ax.scatter(results.loc[results.n > n_threshold].date.dt.dayofyear/30,
               results.loc[results.n > n_threshold, variable], marker='o', ms=1, zorder=0, label='Individual images')
_ = ax.scatter(results_alt.doy/30, results_alt[variable], marker='o', ms=5, zorder=0, color='r', label='Binned by DOY')

ax.format(ylim=(1, 3), xlocator=dr.dayofyear/30, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', ylabel='$\\alpha$', title='Power law slope', fontsize=12, xrotation=45)
ax.legend(loc='lr', ncols=1, ms=15, fontsize=12)

## Panel 2: Length Scale Seasonality
ax = axs[1]

df['month'] = df.datetime.dt.month
df['n'] = df.groupby('datetime').transform(lambda x: len(x))['area']
df['length_scale_km'] = df['area_adj_km']**0.5
idx = df.n > 30
idx = idx & df.area_adj_km.between(41, 90e3)
idx = idx & df.lr_classification
n = df.loc[idx, ['area_adj_km', 'length_scale_km', 'doy']].groupby('doy').count()
med = df.loc[idx, ['area_adj_km', 'length_scale_km', 'doy']].groupby('doy').median()
p90 = df.loc[idx, ['area_adj_km', 'length_scale_km', 'doy']].groupby('doy').quantile(0.90)
p10 = df.loc[idx, ['area_adj_km', 'length_scale_km', 'doy']].groupby('doy').quantile(0.10)
p75 = df.loc[idx, ['area_adj_km', 'length_scale_km', 'doy']].groupby('doy').quantile(0.75)
p25 = df.loc[idx, ['area_adj_km', 'length_scale_km', 'doy']].groupby('doy').quantile(0.25)
var = 'length_scale_km'

ax.plot(med[var], fadedata=[p90[var], p10[var]],
        shadedata=[p75[var], p25[var]], marker='.', label='')
h = []
for alpha, ls, m in zip([1, 0.5, 0.25], ['-', '', ''], ['', 's', 's']):
    h.append(ax.plot([],[],color='tab:blue', alpha=alpha, ls=ls, m=m))
ax.legend(h, ['Median', '25-75%', '10-90%'], ncols=1, loc='ur')

ax.format(ylabel='L (km)', xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', title='Floe length scale', fontsize=12, xrotation=45, xlim=(85, 265))


## Sea ice extent 

ax = axs[2]
for year, group in df_nsidc.groupby(df_nsidc.index.year):
    ax.plot(group.index.dayofyear,
            group['sea_ice_extent'].values/1e3, alpha=0.25, color='tab:blue')
mean_extent = df_nsidc.groupby(df_nsidc.index.dayofyear).mean()
# mean_extent = mean_extent.loc[(mean_extent.index >= dr.dayofyear.min() - 10) & \
#                                  (mean_extent.index <= dr.dayofyear.max() + 10)]
ax.plot(mean_extent.index,
            mean_extent['sea_ice_extent'].values/1e3, alpha=1, lw=3, color='tab:blue')
ax.format(ylabel='Sea ice extent (1000 km$^2$)',  xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', xrotation=45, fontsize=12, xlim=(85, 265), title='Sea ice extent')

## MIZ fraction
ax = axs[3]
for year, group in df_nsidc.groupby(df_nsidc.index.year):
    ax.plot(group.index.dayofyear,
            group['miz_ice_extent'].values/group['sea_ice_extent'].values * 100,
           alpha=0.25, color='tab:blue')
df_nsidc['miz_percent'] = df_nsidc['miz_ice_extent'] / df_nsidc['sea_ice_extent']

mean_percent = df_nsidc.groupby(df_nsidc.index.dayofyear).mean()
ax.plot(mean_percent.index,
            mean_percent['miz_percent'].values*100, alpha=1, lw=3, color='tab:blue')
ax.format(ylabel='MIZ extent (% of ice area)', xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', xrotation=45, fontsize=12, xlim=(85, 265), title='MIZ fraction')

h = []
for alpha, ls, m in zip([1, 0.5, 0.25], ['-', '', ''], ['', 's', 's']):
    h.append(ax.plot([],[],color='tab:blue', alpha=alpha, ls=ls, m=m))
h = [ax.plot([], [], color='tab:blue', lw=1, alpha=1),
     ax.plot([], [], color='tab:blue', lw=3, alpha=1)]
axs[2].legend(h, ['Individual Years', '2003-2020 Mean'], ncols=1,  loc='ur')
axs[3].legend(h, ['Individual Years', '2003-2020 Mean'], ncols=1, loc='ur')

axs.format(abc=True)


# PDF for publication, PNG for slides
fig.save('../figures/fig10_fsd_slope.pdf', dpi=300)
fig.save('../figures/fig10_fsd_slope.png', dpi=300)
