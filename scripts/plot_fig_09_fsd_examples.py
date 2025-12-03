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


#### Example dates and year-to-year variability ####
fig, axs = pplt.subplots(width=7, ncols=2, nrows=2, span=False)

xmax = None
xmin = 41

for ax, example_date in zip(axs, [pd.to_datetime('2017-04-03T13:55:35'),
                                     pd.to_datetime('2012-05-31T11:49:23'),
                                     pd.to_datetime('2005-06-14T13:48:38'),
                                     pd.to_datetime('2007-07-13T13:55:44')]):

    satellite = df.loc[df.datetime == example_date].satellite.values[0]
    n = 0
    for year, group in df.groupby(df.datetime.dt.year):
        data = group.loc[(group.datetime.dt.dayofyear == example_date.dayofyear) & (group.satellite == satellite), 'area_adj_km']
        
        if len(data) > 100:
            fit = powerlaw.Fit(data,xmin=35, xmax=xmax)
            fit.plot_pdf(color='gray', linewidth=1, alpha=0.5, ax=ax)
            fit.plot_ccdf(color='gray', linewidth=1, alpha=0.5, ax=ax)
            n += 1
            
    data = df.loc[(df.datetime.dt.dayofyear == example_date.dayofyear) & (df.satellite==satellite), 'area_adj_km']
    fit = powerlaw.Fit(data,xmin=xmin, xmax=xmax)
    
    fit.plot_pdf(color='k', linewidth=2, ax=ax)
    fit.plot_ccdf(color='k', linewidth=2, ax=ax)
    fit.truncated_power_law.plot_pdf(color='k', ax=ax, linestyle='--', lw=1)
    fit.truncated_power_law.plot_ccdf(color='k', linestyle='--', lw=1, ax=ax)  
    alpha_all = np.round(fit.truncated_power_law.alpha,2)
    se_all =  np.round((alpha_all - 1)/np.sqrt(len(data)),2)
    ax.text(1100, 0.2,'$\\alpha$ = ' + str(alpha_all) + ' $\\pm$' + str(se_all), color='k')

    
    data=df.loc[df.datetime == example_date].area_adj_km
    fit = powerlaw.Fit(data,xmin=35, xmax=xmax)
    alpha_one = np.round(fit.truncated_power_law.alpha,2)
    se_one = np.round((alpha_one - 1)/np.sqrt(len(data)),2)
    ax.text(1100, 0.5,'$\\alpha$ = ' + str(alpha_one) + ' $\\pm$' + str(se_one), color='b')

    
    fit.plot_pdf(color='b', linewidth=2, ax=ax)
    fit.truncated_power_law.plot_pdf(color='b', ax=ax, linestyle='--', lw=1)
    fit.plot_ccdf(color='b', linewidth=2, ax=ax)
    fit.truncated_power_law.plot_ccdf(color='b', linestyle='--', lw=1, ax=ax)
    ax.format(xlabel = 'Floe area (km$^2$)', ylabel='Probability',
             xlim=(30, 9000), title='Day of Year ' + str(example_date.dayofyear) + ' (' + example_date.strftime('%b') + ')')
    
    
    ax.text(150, 0.4, 'CCDF')
    ax.text(150, 0.004, 'PDF')


    # ax.text(750,.55,'xmin = '+str(fit.xmin)+' km$^2$')
    # ax.text(750,.3,'xmax = '+str(np.round(np.amax(data),0))+' km$^2$')
    # ax.text(750,.18,'N = '+str(len(data[data > fit.xmin])))
    h = [ax.plot([],[],color=c, lw=lw, ls=ls) for c, lw, ls in zip(['b', 'b',  'k', 'k', 'gray'],
                                                                   [2, 1, 2, 1, 2, 1],
                                                                   ['-', '--', '-', '--', '-', '-'])]
    ax.legend(h, [example_date.strftime('%Y-%m-%d' + ' (Empirical)'),
               example_date.strftime('%Y-%m-%d' + ' (Fitted)'),
               'Binned 2003-20 (Empirical)',
                  'Binned 2003-20 (Fitted)',
                  'Individual Years (n=' + str(n) + ')'], loc='ll', order='F', ncols=1)

axs.format(abc=True, fontsize=10)
fig.save('../figures/fig09_example_dates_fsd.pdf')
fig.save('../figures/fig09_example_dates_fsd.png', dpi=300)
