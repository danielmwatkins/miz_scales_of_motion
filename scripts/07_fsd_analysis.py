#Import everything
import pandas as pd
import numpy as np
import powerlaw
import os
import proplot as pplt

data_loc='../data/floe_tracker/ift_floe_property_tables/clean/'
flist=[a for a in os.listdir(data_loc) if a.startswith('ift_clean')]
dflist = []
for fname in flist:
    dflist.append(pd.read_csv(data_loc + fname))
df=pd.concat(dflist)
df=df.reset_index()
df['perim_km']=df.perimeter*.25
df['area_km']=df.area*.25*.25
df['circ']=( (4*np.pi * df['area_km'])/(df['perim_km']**2))
df['doy']=[int(datetime.fromisoformat(a).strftime('%j')) for a in df.datetime.values]
df['year']=[int(datetime.fromisoformat(a).strftime('%Y')) for a in df.datetime.values]
# df.to_csv(data_loc + 'df_allfloes.csv')


### Compute slope of power law for images with sufficient number of floes
results = []
for year, df_year in df.groupby('year'):
    for date, df_date in df_year.groupby('datetime'):
        if len(df_date) > 50: # Including all with at least 50, however, later figure only uses those with at least 100
            area = df_date.area_km # adjustment for the over dilation
            fit = powerlaw.Fit(area, xmin=10)
            alpha = fit.power_law.alpha
            comp_powerlaw = fit.distribution_compare('lognormal', 'power_law')[0]
            comp_exponential = fit.distribution_compare('lognormal', 'exponential')[0]
            comp_pl_v_exp = fit.distribution_compare('power_law', 'exponential')[0]
            results.append([date, len(df_date), alpha, comp_powerlaw, comp_exponential, comp_pl_v_exp])
results = pd.DataFrame(results, columns=['date', 'n', 'alpha', 'lr_LN_v_PL', 'lr_LN_v_EXP', 'lr_PL_v_EXP'])        
results['doy'] = results.date.dt.dayofyear
results['month'] = results.date.dt.month
doy = results.groupby('month').mean()['doy']

### Same analysis but binning by day of year 
results_alt = []
for date, df_date in df.groupby(df.datetime.dt.dayofyear):
    if len(df_date) > 50:
        area = df_date.area_km # adjustment for the over dilation
        fit = powerlaw.Fit(area, xmin=10)
        alpha = fit.power_law.alpha
        comp_powerlaw = fit.distribution_compare('lognormal', 'power_law')[0]
        comp_exponential = fit.distribution_compare('lognormal', 'exponential')[0]
        comp_pl_v_exp = fit.distribution_compare('power_law', 'exponential')[0]
        results_alt.append([date, len(df_date), alpha, comp_powerlaw, comp_exponential, comp_pl_v_exp])
results_alt = pd.DataFrame(results_alt, columns=['date', 'n', 'alpha', 'lr_LN_v_PL', 'lr_LN_v_EXP', 'lr_PL_v_EXP'])        
results_alt['doy'] = results_alt.date.values
# Convert DOY to month based on a non-leap year
results_alt['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(results_alt.doy, 'D')).dt.month


fig, ax = pplt.subplots(width=4)
min_n = 100
time_idx = (results.month != 3) & (results.month <= 9)
df_pivot = results.loc[time_idx & (results.n > min_n)].pivot_table(index='date', values='alpha', columns='month')
month_counts = df_pivot.notnull().sum() 
month_idx = [m for m in month_counts.index if month_counts[m] > 100]
_ = ax.box(doy[month_idx].values/30, df_pivot[month_idx], facecolor='w', alpha=0.85, zorder=1, marker='')
_ = ax.scatter(results.loc[results.n > min_n].date.dt.dayofyear/30, results.loc[results.n > min_n].alpha, marker='o', ms=1, zorder=0, label='Individual images')
_ = ax.scatter(results_alt.doy/30, results_alt.alpha, marker='o', ms=5, zorder=0, color='r', label='Binned by day of year')

ax.format(ylim=(1.4, 1.6), xlocator=dr.dayofyear/30, xformatter=[d.strftime('%b') for d in dr],
          xlabel='', ylabel='$\\alpha$', title='Estimated $\\alpha$ by time of year', fontsize=12)
ax.legend(loc='lr', ncols=1, ms=10)