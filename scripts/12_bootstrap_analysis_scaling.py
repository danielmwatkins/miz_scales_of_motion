"""Calculate the slopes and confidence intervals for the log-normal MLE method and for the linear regression method."""

"""calculate the best-fit model parameters and uncertainty for the deformation scaling."""

from scipy.optimize import curve_fit
import scipy.stats as stats
import pandas as pd
import proplot as pplt
import numpy as np
import warnings
warnings.simplefilter('ignore')

##### Load deformation data #######
df = pd.read_csv('../data/deformation/sampled_results.csv', index_col=0, parse_dates=['datetime'])
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

##### Helper functions ######
def normal_log_likelihood(eps, L, beta):
    n = len(eps)
    data = np.log(eps*L**beta)
    mu = np.mean(data)
    sigma = np.std(data)
    normalizer = -n/2*np.log(2*np.pi*sigma**2)
    return normalizer - np.sum(data - mu)**2/(2*sigma**2)

def get_cv(n):
    dist = stats.t(n-1)        
    return dist.ppf(0.975)
      
def f(x, m, b):
    """Order 1 line"""
    return m*x + b

def rms(y, yfit):
    return np.sqrt(np.sum((y - yfit) ** 2))

##### Get stratified sample by log bin #####
rs = 32413
n = 400 # Had done 500 before.
samples = {4: [], 5: [], 6: []}

# for (month, log_bin), group in df.loc[df['unique_floes_sample'], :].groupby(['month', 'log_bin']):         
for (month, log_bin), group in df.loc[df['no_overlap_sample'], :].groupby(['month', 'log_bin']):         
    if len(group) >= n:
        if month in samples:
            # weight by the number of observations from that date
            w = 1 / group.groupby('datetime').transform(lambda x: len(x))['triangle_number']
            samples[month].append(group.sample(n, replace=False, weights=w, random_state=rs + month + log_bin))
    else:
        pass

for month in samples:
    samples[month] = pd.concat(samples[month], axis=0)

###### Calculate beta #######
# 1. Bootstrap analysis to get uncertainty of slopes
bs_table = []
for month in samples:
    strat_samp = samples[month]    
    mle_results = []
    lsq_results = []
    for repeat in range(1000):
        resamp = strat_samp.sample(len(strat_samp), replace=True, random_state=rs + repeat)

        # MLE method - could adjust the function to accept the same data
        likelihood_results = pd.Series(np.nan, index=np.linspace(0.01, 1, 200))
        for beta in likelihood_results.index:
            likelihood_results.loc[beta] = normal_log_likelihood(resamp['total_deformation']*(60*60*24),
                                                                 resamp['L'], beta)
            
        # if the idxmax() is equal to one side or the other, then it didn't find a maximum.
        first = np.abs(likelihood_results.idxmax() - likelihood_results.values[0]) < 1e-5
        last = np.abs(likelihood_results.idxmax() - likelihood_results.values[-10]) < 1e-5
        if not (first | last):
            mle_results.append(likelihood_results.idxmax())
        del likelihood_results

        # LSQ method - compute bin mean, variance, then fit curve with weights
        resamp_mean = resamp.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').mean()   
        SE = resamp.groupby('log_bin').apply(lambda x: stats.sem(x['total_deformation']))
        
        # compute the standard error of the mean
        n_bin = resamp.groupby('log_bin').count()['L']
        cv = np.array([get_cv(n) for n in n_bin])
        y = np.log(resamp_mean['total_deformation'].values*24*60*60)
        x = np.log(resamp_mean['L'].values)
        sigma = np.log([cv[i] * SE.values[i]*24*60*60 for i in range(len(cv))])

        p0 = 3, 1 # initialize curve fit 
        popt, pcov = curve_fit(f, x, y, p0, sigma=sigma, absolute_sigma=True)
        yfit = f(x, *popt)
      
        lsq_results.append(-popt[0])

    # Get confidence interval through percentile method
    q1_mle, q2_mle = np.quantile(np.array(mle_results), [0.025, 0.975])
    q1_lsq, q2_lsq = np.quantile(np.array(lsq_results), [0.025, 0.975])
    bs_table.append([month, q1_mle, q2_mle, q1_lsq, q2_lsq])
    
bs_table = pd.DataFrame(bs_table,
                        columns=['month',
                                 'min_beta_mle', 'max_beta_mle',
                                 'min_beta_lsq', 'max_beta_lsq']).set_index('month')
bs_table['beta_mle'] = np.nan
bs_table['beta_lsq'] = np.nan

# Calculation of slopes over stratified sample
for month in samples:
    strat_samp = samples[month]
    likelihood_results = pd.Series(np.nan, index=np.linspace(0.01, 1, 200))
    for beta in likelihood_results.index:
        likelihood_results.loc[beta] = normal_log_likelihood(strat_samp['total_deformation']*(60*60*24),
                                                             strat_samp['L'], beta)
    beta = likelihood_results.idxmax()
    bs_table.loc[month, 'beta_mle'] = beta 

    samp_mean = strat_samp.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').mean()   
    SE = strat_samp.groupby('log_bin').apply(lambda x: stats.sem(x['total_deformation']))
    
    # compute the standard error of the mean
    n = 400
    cv = get_cv(n)
    y = np.log(samp_mean['total_deformation'].values*24*60*60)
    x = np.log(samp_mean['L'].values)
    sigma = np.log(cv * SE.values*24*60*60)
    
    
    p0 = 3, 1 # initialize curve fit 
    popt, pcov = curve_fit(f, x, y, p0, sigma=sigma, absolute_sigma=True)
    yfit = f(x, *popt)
    beta, intercept = popt
    
    bs_table.loc[month, 'beta_lsq'] = -beta
    bs_table.loc[month, 'a_lsq'] = np.exp(intercept)

bs_table.to_csv('../data/deformation/scaling_estimates.csv')