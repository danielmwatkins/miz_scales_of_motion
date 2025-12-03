"""
Plotting seasonality of the FSD
"""
import os
import pandas as pd
import ultraplot as pplt
import numpy as np
import powerlaw

#### Seasonality of the FSD ####
results = []
for year in range(2003, 2021):
    results.append(pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_{y}.csv'.format(y=year), parse_dates=['date']))
results = pd.concat(results, axis=0)
results_alt = pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_all_years_by_DOY.csv')
dr = pd.date_range('2020-04-01', '2020-09-01', freq='1MS')

doy = results.groupby('month').mean()['doy']
fig, ax = pplt.subplots(width=4)
n_threshold = 1 #
month_threshold = 300
variable = 'alpha_tpl'
time_idx = (results.month != 3) & (results.month <= 9)
df_pivot = results.loc[time_idx & (results.n > n_threshold)].pivot_table(index='date', values=variable, columns='month')
month_counts = df_pivot.notnull().sum() 
month_idx = [m for m in month_counts.index if month_counts[m] > month_threshold]
_ = ax.box(doy[month_idx].values/30, df_pivot[month_idx], facecolor='w', alpha=0.7, zorder=1, lw=1.5, marker='')
_ = ax.scatter(results.loc[results.n > n_threshold].date.dt.dayofyear/30,
               results.loc[results.n > n_threshold, variable], marker='o', ms=1, zorder=0, label='Individual images')
_ = ax.scatter(results_alt.doy/30, results_alt[variable], marker='o', ms=5, zorder=0, color='r', label='Binned by day of year')

ax.format(ylim=(1, 3), xlocator=dr.dayofyear/30, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', ylabel='$\\alpha$', title='Seasonality of power law slope', fontsize=12)
ax.legend(loc='lr', ncols=1, ms=10, fontsize=12)

# PDF for publication, PNG for slides
fig.save('../figures/fig10_fsd_slope.pdf', dpi=300)
fig.save('../figures/fig10_fsd_slope.png', dpi=300)


#### TO DO ####
# Adding the other seasonality figures would help
# MIZ extent vs sea ice extent
# MIZ vs seasonal ice zone -- is it that the seasonal ice zone becomes more MIZ like?
# Average floe size (mean floe length scale, range)
# 
