"""Identify the appropriate min and sample size via experiments"""

import pandas as pd
import numpy as np
import powerlaw
import os
import warnings
warnings.simplefilter('ignore')
import warnings
warnings.simplefilter('ignore')

print("Loading data")
temp = []
for year in range(2003, 2021):
    temp.append(pd.read_csv('../data/floe_tracker/ift_floe_property_tables/with_nsidc/ift_floe_properties_{y}.csv'.format(y=year),
                            parse_dates=True))
df = pd.concat(temp)
df['datetime'] = pd.to_datetime(df['datetime'].values)
df = df.loc[(df.datetime.dt.dayofyear >= 91) & (df.datetime.dt.dayofyear <= 258)]

df['perim_km'] = df.perimeter*.25
df['area_km'] = df.area*.25*.25
df['area_adj_km'] = (np.sqrt(df.area) + 8)**2*.25*.25 # 6 pixel shift minimizes error against manual
df['doy'] = df.datetime.dt.dayofyear
df['year'] = df.datetime.dt.year

df_filtered = df.groupby('datetime').filter(lambda x: len(x) > 200)
date_sample = np.random.choice(np.unique(df_filtered.datetime), 100)

xmin_vals = np.arange(1, 75, 2)
results_pl = []
results_ln = [] 
results_tpl = []


# Compute variation in the fit quality as a function of the xmin
# This takes about three minutes
print("Computing variation in fit as a function of x min")
for count, date in enumerate(date_sample):
    if (count % 10) == 0:
        print(count, date)
        
    area = df.loc[df.datetime == date, 'area_adj_km']
    temp_pl = []
    temp_ln = []
    temp_tpl = []
    for x in xmin_vals:
        fit = powerlaw.Fit(area, xmin=x)
        temp_pl.append(fit.power_law.D)
        temp_tpl.append(fit.truncated_power_law.D)
        temp_ln.append(fit.lognormal.D)
    results_pl.append(pd.Series(temp_pl, index=xmin_vals))
    results_ln.append(pd.Series(temp_ln, index=xmin_vals))
    results_tpl.append(pd.Series(temp_tpl, index=xmin_vals))   
    
large_groups = df.groupby('datetime').filter(lambda x: len(x) > 300)
dates = np.unique(large_groups['datetime'])
sample = np.random.choice(dates, 500)

# This is slow: takes about 17 minutes
relative_error = {}
thresholds = np.arange(30, 301, 10)
n_resamples = 100
count = 0
xmin = 35
print("Computing error in alpha with varying sample size")
for threshold in thresholds:
    print(threshold)
    relative_error[threshold] = []
    for date in sample:
        sample_group = large_groups.loc[large_groups.datetime == date]
        area = sample_group.area_adj_km
        fit = powerlaw.Fit(area, xmin=xmin)
        alpha = fit.power_law.alpha
        
        for ii in range(n_resamples):
            area_resamp = sample_group.sample(threshold).area_adj_km
            fit = powerlaw.Fit(area_resamp, xmin=xmin)
            relative_error[threshold].append((fit.power_law.alpha - alpha) / alpha)

pd.DataFrame(relative_error).to_csv('../data/test_data/relative_error_pl_alpha.csv')
relative_error = pd.DataFrame(relative_error)

test_tpl = pd.DataFrame(results_tpl, index=date_sample)
test_pl = pd.DataFrame(results_pl, index=date_sample)
test_ln = pd.DataFrame(results_ln, index=date_sample)

test_tpl.to_csv('../data/test_data/ks_data_tpl.csv')
test_pl.to_csv('../data/test_data/ks_data_pl.csv')
test_ln.to_csv('../data/test_data/ks_data_ln.csv')
