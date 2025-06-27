"""
Plotting code for
fig05_fsd_setup_figures
fig07_likelihood_ratio_results
fig09_all_floes_v_tracked_FSD
"""


import os
import pandas as pd
import proplot as pplt
import numpy as np
import powerlaw

#### Model evaluation
test_tpl = pd.read_csv('../data/test_data/ks_data_tpl.csv', index_col=0)
test_pl = pd.read_csv('../data/test_data/ks_data_pl.csv', index_col=0)
test_ln = pd.read_csv('../data/test_data/ks_data_ln.csv', index_col=0)

test_tpl.columns = test_tpl.columns.astype(int)
test_pl.columns = test_pl.columns.astype(int)
test_ln.columns = test_ln.columns.astype(int)

relative_error = pd.read_csv('../data/test_data/relative_error_pl_alpha.csv', index_col=0)
relative_error.columns = relative_error.columns.astype(int)
fig, axs = pplt.subplots(ncols=4, nrows=1, sharex=False, sharey=False)

for ax, data, title in zip(axs, [test_pl, test_tpl, test_ln], ['Power Law', 'Truncated Power Law', 'Lognormal']):
    ax.plot(data.mean(axis=0), shadedata=data.std(axis=0), fadedata=[data.min(axis=0), data.max(axis=0)], label='Mean', color='tab:blue')
    ax.plot([],[], lw=4, alpha=0.5, label='Std. Dev.', color='tab:blue')
    ax.plot([],[], lw=4, alpha=0.25, label='Min-Max', color='tab:blue')
    ax.format(xlabel='Value of x-min (km$^2$)', ylabel='KS Statistic', title='{t}'.format(t=title), ylim=(0, 0.6))
    ax.legend(loc='ur', ncols=1)
    ax.axvline(35, ls='--')
axs.format(abc=False)

ax = axs[3]
ax.plot(relative_error.median(axis=0),
        shadedata=[relative_error.quantile(0.25, axis=0),
                   relative_error.quantile(0.75, axis=0)],
        fadedata=[relative_error.quantile(0.1, axis=0),
                  relative_error.quantile(0.9, axis=0)], color='tab:blue')
ax.format(ylocator=[-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15], ylim=(-0.15, 0.15),
          yformatter=['-15%', '-10%', '-5%', '0%', '5%', '10%', '15%'], xlabel='Minimum number of floes')
ax.format(title='Sampling error')


h = []
for alpha, ls, m in zip([1, 0.5, 0.25], ['-', '', ''], ['', 's', 's']):
    h.append(ax.plot([],[],color='tab:blue', alpha=alpha, ls=ls, m=m))
axs[3].legend(h, ['Median', '25-75%', '10-90%'], ncols=1, loc='ur')
axs[3].legend(h, ['Median', '25-75%', '10-90%'], ncols=1, loc='ur')

axs.format(abc=True, fontsize=12)
fig.save('../figures/fig05_fsd_setup_figures.pdf', dpi=300)
fig.save('../figures/fig05_fsd_setup_figures.png', dpi=300)

#### Log likelihood results
results = []
for year in range(2003, 2021):
    results.append(pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_{y}.csv'.format(y=year), parse_dates=['date']))
results = pd.concat(results, axis=0)
results_alt = pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_all_years_by_DOY.csv')

alpha = 0.05
dr = pd.date_range('2020-04-01', '2020-09-01', freq='1MS')

fig, axs = pplt.subplots(ncols=3, nrows=2, span=False)
for ax, comp in zip(axs[0,:], ['PL_v_TPL', 'PL_v_LN', 'LN_v_TPL']):

    # False discovery rate adjustment - more stringent p-value than alpha since we 
    # are calculating it numerous times
    p = list(results['lr_' + comp])
    p.sort()
    test = np.array(p) < alpha*(np.arange(1, len(p)+1)/len(p))
    alpha_equiv = max(np.array(p)[test])
    ax.scatter(results.doy, results['lr_' + comp].where(results['p_' + comp] > alpha_equiv),
               marker='o', facecolor='gray', edgecolor='k')
    ax.scatter(results.doy, results['lr_' + comp].where(results['p_' + comp] <= alpha_equiv),
               marker='o', facecolor='blue2', edgecolor='k')
    
    ax.format(ylabel='Log likelihood ratio', title=comp.replace('v', 'vs.').replace('_', ' '),
              xlabel='', ylim=(-65, 65))
    
    ax.format(urtitle='Favors ' + comp.split('_')[0],
              lrtitle='Favors ' + comp.split('_')[2])
    
for ax, comp in zip(axs[1,:], ['PL_v_TPL', 'PL_v_LN', 'LN_v_TPL']):
    p = list(results_alt['lr_' + comp])
    p.sort()
    test = np.array(p) < alpha*(np.arange(1, len(p)+1)/len(p))
    alpha_equiv = max(np.array(p)[test])

    ax.scatter(results_alt.doy, results_alt['lr_' + comp].where(results_alt['p_' + comp] > alpha_equiv),
               marker='o', facecolor='gray', edgecolor='k')
    
    ax.scatter(results_alt.doy, results_alt['lr_' + comp].where(results_alt['p_' + comp] <= alpha_equiv),
               marker='o', facecolor='green2', edgecolor='k')
    
    ax.format(ylabel='Log likelihood ratio', title=comp.replace('v', 'vs.').replace('_', ' '),
             xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr], xlabel='',
              ylim=(-65, 65), xlim=(80, 250))

    ax.format(urtitle='Favors ' + comp.split('_')[0],
              lrtitle='Favors ' + comp.split('_')[2])

axs.format(leftlabels=['Single image', 'Binned by DOY'], abc=True)    
fig.save('../figures/fig07_likelihood_ratio_results.png', dpi=300)
fig.save('../figures/fig07_likelihood_ratio_results.pdf', dpi=300)

#### Low SIC vs High SIC ####
# Placeholder

#### All vs Tracked Floes ####
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


fig, ax = pplt.subplots(width=4)
xmax=90000
for year, group in df.groupby(df.datetime.dt.year):
    data = group.area_adj_km
    fit = powerlaw.Fit(data,xmin=35, xmax=xmax)
    fit.plot_pdf(color='k', linewidth=0.7, ax=ax, label='', alpha=0.3)
    fit.plot_ccdf(color='k', linewidth=0.7, ax=ax, alpha=0.3)
data = df.area_adj_km
fit = powerlaw.Fit(data,xmin=35, xmax=xmax)
fit.plot_pdf(color='k', linewidth=2, ax=ax, label='All floes')
fit.plot_ccdf(color='k', linewidth=2, ax=ax)

for year, group in df.loc[(df.floe_id != 'unmatched')].groupby(
                df.loc[(df.floe_id != 'unmatched')].datetime.dt.year):
    data = group.area_adj_km
    fit = powerlaw.Fit(data,xmin=35, xmax=xmax)
    fit.plot_pdf(color='r', linewidth=0.7, ax=ax, label='', alpha=0.3, ls='--')
    fit.plot_ccdf(color='r', linewidth=0.7, ax=ax, alpha=0.3, ls='--')

data = df.loc[(df.floe_id != 'unmatched')].area_adj_km
fit = powerlaw.Fit(data,xmin=35, xmax=xmax)
fit.plot_pdf(color='r', linewidth=2, ax=ax, label='Tracked only', ls='--')
fit.plot_ccdf(color='r', linewidth=2, ax=ax, ls='--')
ax.legend(loc='ll', ncols=1)
ax.format(xlabel='Floe area (km$^2$)', ylabel='Probability')
fig.save('../figures/fig09_all_floes_v_tracked_FSD.png')
fig.save('../figures/fig09_all_floes_v_tracked_FSD.pdf')