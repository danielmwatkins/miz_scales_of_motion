import os
import pandas as pd
import ultraplot as pplt
import numpy as np
import powerlaw

#### Log likelihood results
results = []
for year in range(2003, 2021):
    results.append(pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_{y}.csv'.format(y=year), parse_dates=['date']))
results = pd.concat(results, axis=0)
results_alt = pd.read_csv('../data/floe_tracker/ift_fsd_tables/ift_fsd_table_all_years_by_DOY.csv')

alpha = 0.05
dr = pd.date_range('2020-04-01', '2020-09-01', freq='1MS')

fig, axs = pplt.subplots(ncols=3, nrows=2, span=False, sharex=False)
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
              xlabel='', ylim=(-65, 65), xlocator=dr.dayofyear, xformatter=[d.strftime('%b') for d in dr])
    
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
fig.save('../figures/fig08_likelihood_ratio_results.png', dpi=300)
fig.save('../figures/fig08_likelihood_ratio_results.pdf', dpi=300)
