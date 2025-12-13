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
dr_doy = results.groupby('month').mean()['doy']

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

###### Compute seasonality for pack/miz and edge/coast by DOY
#### Move this to precalculate later #####
n_threshold = 100
edge_results = []
coast_results = []
x_min = 41
df['doy'] = df.datetime.dt.dayofyear

for temp_results, idx_sel in zip([edge_results, coast_results],
                            [df.edge_dist_km.between(0, 100),
                             df.coast_dist_km.between(0, 100)]):
    for doy, df_doy in df.loc[idx_sel, :].groupby('doy'):
        for satellite, group in df_doy.groupby('satellite'):
            if len(group.loc[group.area_adj_km > x_min]) > n_threshold:
                area = group.area_adj_km
                fit = powerlaw.Fit(area, xmin=x_min, verbose=False)
                temp_results.append([doy, satellite, len(df_doy),
                                    fit.power_law.alpha,
                                    fit.truncated_power_law.alpha,
                                    fit.truncated_power_law.parameter2,
                                    fit.power_law.D,
                                    fit.truncated_power_law.D
                                ])
edge_results = pd.DataFrame(edge_results, columns=['doy', 'satellite', 'n',
                                         'alpha', 'alpha_tpl', 'lambda_tpl',
                                         'D_PL', 'D_TPL'])        
edge_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(edge_results.doy, 'D')).dt.month      
edge_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(edge_results.doy, 'D')).dt.month

coast_results = pd.DataFrame(coast_results, columns=['doy', 'satellite', 'n',
                                         'alpha', 'alpha_tpl', 'lambda_tpl',
                                         'D_PL', 'D_TPL'])        
coast_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(coast_results.doy, 'D')).dt.month      
coast_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(coast_results.doy, 'D')).dt.month

miz_results = []
pack_results = []
df['doy'] = df.datetime.dt.dayofyear

for temp_results, idx_sel in zip([miz_results, pack_results],
                            [df.nsidc_sic.between(0.15, 0.85),
                             df.nsidc_sic.between(0.85, 1.0)]):
    for doy, df_doy in df.loc[idx_sel, :].groupby('doy'):
        for satellite, group in df_doy.groupby('satellite'):
            if len(group.loc[group.area_adj_km > x_min]) > n_threshold:
                area = group.area_adj_km
                fit = powerlaw.Fit(area, xmin=x_min, verbose=False)
                temp_results.append([doy, satellite, len(df_doy),
                                    fit.power_law.alpha,
                                    fit.truncated_power_law.alpha,
                                    fit.truncated_power_law.parameter2,
                                    fit.power_law.D,
                                    fit.truncated_power_law.D
                                ])

miz_results = pd.DataFrame(miz_results, columns=['doy', 'satellite', 'n',
                                         'alpha', 'alpha_tpl', 'lambda_tpl',
                                         'D_PL', 'D_TPL'])        
miz_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(miz_results.doy, 'D')).dt.month      
miz_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(miz_results.doy, 'D')).dt.month

pack_results = pd.DataFrame(pack_results, columns=['doy', 'satellite', 'n',
                                         'alpha', 'alpha_tpl', 'lambda_tpl',
                                         'D_PL', 'D_TPL'])        
pack_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(pack_results.doy, 'D')).dt.month      
pack_results['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(pack_results.doy, 'D')).dt.month

fig, axs =pplt.subplots(ncols=3, nrows=1, share=False) 
ax = axs[0]
n_threshold = 1 # check what this one is doing -- is it just to see if the month has anything in it?
month_threshold = 300
variable = 'alpha_tpl'
time_idx = (results.month != 3) & (results.month <= 9)
df_pivot = results.loc[time_idx & (results.n > n_threshold)].pivot_table(index='date', values=variable, columns='month')
month_counts = df_pivot.notnull().sum() 
month_idx = [m for m in month_counts.index if month_counts[m] > month_threshold]
_ = ax.box(dr_doy[month_idx].values/30, df_pivot[month_idx], facecolor='w', alpha=0.7, zorder=1, lw=1.5, marker='')
_ = ax.scatter(results.loc[results.n > n_threshold].date.dt.dayofyear/30,
               results.loc[results.n > n_threshold, variable], marker='o', ms=1, zorder=0, label='Individual images')
_ = ax.scatter(results_alt.doy/30, results_alt[variable], marker='o', ms=5, zorder=0, color='r', label='Binned by DOY')
ax.legend(loc='lr', ncols=1, ms=15, fontsize=12)

ax = axs[1]
_ = ax.scatter(miz_results.doy/30, miz_results[variable].values, marker='o', ms=5, zorder=0, color='slateblue', label='MIZ')
_ = ax.scatter(pack_results.doy/30, pack_results[variable].values, marker='o', ms=5, zorder=0, color='tab:green', label='Pack Ice')

ax.legend(loc='lr', ncols=1, ms=15, fontsize=12)
for ax in axs:
    ax.format(ylim=(1, 3), xlocator=dr.dayofyear/30, xlim=(2.9, 9),
              xformatter=[d.strftime('%b') for d in dr],
               abc=True, xlabel='', ylabel='$\\alpha$', fontsize=12, xrotation=0)

ax = axs[2]
_ = ax.scatter(edge_results.doy/30, edge_results[variable].values, marker='o', ms=5, zorder=0, color='tab:blue', label='Near Edge')
_ = ax.scatter(coast_results.doy/30, coast_results[variable].values, marker='o', ms=5, zorder=0, color='tab:orange', label='Near Coast')

ax.legend(loc='lr', ncols=1, ms=15, fontsize=12)
for ax in axs:
    ax.format(ylim=(1, 3), xlocator=dr.dayofyear/30, xlim=(2.9, 9),
              xformatter=[d.strftime('%b') for d in dr],
               abc=True, xlabel='', ylabel='$\\alpha$', fontsize=12, xrotation=0)

axs[0].format(title='All floes')
axs[1].format(title='SIC partition')
axs[2].format(title='Edge and coast distance')

# PDF for publication, PNG for slides
for imtype in ['pdf', 'png']:
    fig.save('../figures/{im}/fig12_fsd_seasonality.{im}'.format(im=imtype), dpi=300)

