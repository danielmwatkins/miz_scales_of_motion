"""Calculate the FSD fit for each image and grouped by day of year, then save results in the ift_fsd_tables folder"""

import pandas as pd
import numpy as np
import powerlaw
import os

x_min = 35
n_threshold = 100
saveloc = '../data/floe_tracker/ift_fsd_tables/'
dataloc = '../data/floe_tracker/'
# join these from yearly files instead
floe_lib_clean = {}
for file in os.listdir('../data/floe_tracker/ift_floe_property_tables/clean/'):
    if 'csv' in file: 
        year = int(file.replace('.csv', '').split('_')[-1])
        floe_lib_clean[year] = pd.read_csv('../data/floe_tracker/ift_floe_property_tables/clean/' + file,
                                         index_col=0, dtype={'classification': str})
        floe_lib_clean[year]['datetime'] = pd.to_datetime(floe_lib_clean[year]['datetime'])

df = pd.concat(floe_lib_clean).reset_index()
# df = pd.read_csv(dataloc + 'ift_floe_properties.csv', parse_dates=True)
# df['datetime'] = pd.to_datetime(df['datetime'].values)

df = df.loc[(df.datetime.dt.dayofyear >= 91) & (df.datetime.dt.dayofyear <= 258)]

df['perim_km'] = df.perimeter*.25
df['area_km'] = df.area*.25*.25
df['area_adj_km'] = (np.sqrt(df.area) + 8)**2*.25*.25 # Pixel shift minimizes error against manual
df['doy'] = df.datetime.dt.dayofyear
df['year'] = df.datetime.dt.year

for year, df_year in df.groupby('year'):
    print(year)
    results = []
    for date, df_date in df_year.groupby('datetime'):
        if len(df_date.loc[df_date.area_adj_km > x_min]) > n_threshold:
            area = df_date.area_adj_km
            fit = powerlaw.Fit(area, xmin=x_min, verbose=False)
            lr_PL_v_TPL, p_PL_v_TPL = fit.distribution_compare('power_law', 'truncated_power_law', nested=True)
            lr_PL_v_LN, p_PL_v_LN = fit.distribution_compare('power_law', 'lognormal')
            lr_LN_v_TPL, p_LN_v_TPL = fit.distribution_compare('lognormal', 'truncated_power_law')
        
            results.append([date, len(df_date),
                            fit.power_law.alpha,
                            fit.truncated_power_law.alpha,
                            fit.truncated_power_law.parameter2,
                            fit.lognormal.mu,
                            fit.lognormal.sigma,
                            fit.power_law.D,
                            fit.truncated_power_law.D,
                            fit.lognormal.D,
                            lr_PL_v_TPL,
                            p_PL_v_TPL,
                            lr_PL_v_LN,
                            p_PL_v_LN,
                            lr_LN_v_TPL,
                            p_LN_v_TPL
                            ])
    results = pd.DataFrame(results, columns=['date', 'n',
                                             'alpha', 'alpha_tpl', 'lambda_tpl', 'mu', 'sigma',
                                             'D_PL', 'D_TPL', 'D_LN',
                                             'lr_PL_v_TPL', 'p_PL_v_TPL', 'lr_PL_v_LN', 'p_PL_v_LN', 'lr_LN_v_TPL', 'p_LN_v_TPL'])        
    results['doy'] = results.date.dt.dayofyear
    results['month'] = results.date.dt.month
    import os
    os.listdir('../data/')
    results.to_csv(saveloc + 'ift_fsd_table_{y}.csv'.format(y=year))


# Same thing grouped by DOY
results_alt = []
for doy, df_doy in df.groupby(df.datetime.dt.dayofyear):
    for satellite, group in df_doy.groupby('satellite'):
        if len(group.loc[group.area_adj_km > x_min]) > n_threshold:
            area = group.area_adj_km
            fit = powerlaw.Fit(area, xmin=x_min, verbose=False)
            lr_PL_v_TPL, p_PL_v_TPL = fit.distribution_compare('power_law', 'truncated_power_law', nested=True)
            lr_PL_v_LN, p_PL_v_LN = fit.distribution_compare('power_law', 'lognormal')
            lr_LN_v_TPL, p_LN_v_TPL = fit.distribution_compare('lognormal', 'truncated_power_law')
            results_alt.append([doy, satellite, len(df_doy),
                                fit.power_law.alpha,
                                fit.truncated_power_law.alpha,
                                fit.truncated_power_law.parameter2,
                                fit.lognormal.mu,
                                fit.lognormal.sigma,
                                fit.power_law.D,
                                fit.truncated_power_law.D,
                                fit.lognormal.D,
                                lr_PL_v_TPL,
                                p_PL_v_TPL,
                                lr_PL_v_LN,
                                p_PL_v_LN,
                                lr_LN_v_TPL,
                                p_LN_v_TPL
                            ])
results_alt = pd.DataFrame(results_alt, columns=['doy', 'satellite', 'n',
                                         'alpha', 'alpha_tpl', 'lambda_tpl', 'mu', 'sigma',
                                         'D_PL', 'D_TPL', 'D_LN',
                                         'lr_PL_v_TPL', 'p_PL_v_TPL', 'lr_PL_v_LN', 'p_PL_v_LN', 'lr_LN_v_TPL', 'p_LN_v_TPL'])        
        
results_alt['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(results_alt.doy, 'D')).dt.month      
        
results_alt['month'] = (pd.to_datetime('2001-01-01') + pd.to_timedelta(results_alt.doy, 'D')).dt.month
results_alt.to_csv(saveloc + 'ift_fsd_table_all_years_by_DOY.csv')