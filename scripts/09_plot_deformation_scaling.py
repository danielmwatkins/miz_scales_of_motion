"""calculate the best-fit model parameters and uncertainty for the deformation scaling."""

from scipy.stats import linregress
import pandas as pd
import proplot as pplt
import numpy as np

def normal_log_likelihood(eps, L, beta):
    n = len(eps)
    data = np.log(eps*L**beta)
    mu = np.mean(data)
    sigma = np.std(data)
    normalizer = -n/2*np.log(2*np.pi*sigma**2)
    return normalizer - np.sum(data - mu)**2/(2*sigma**2)

##### Load deformation data #######
df = pd.read_csv('../data/deformation/sampled_results.csv', index_col=0, parse_dates=['datetime'])
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

###### Calculating #######

# Stratified sample by log bin
rs = 32413
n = 500
samples = {4: [], 5: [], 6: []}
for (month, log_bin), group in df.loc[df['no_overlap_sample'], :].groupby(['month', 'log_bin']):         
    if len(group) >= n:
        if month in samples:
            # weight by the number of observations from that date
            w = 1 / group.groupby('datetime').transform(lambda x: len(x))['triangle_number']
            samples[month].append(group.sample(n, replace=False, weights=w, random_state=rs + month + log_bin))
            # weight this by the number of observations per day within each bin
    else:
        pass
        
        # print(month, log_bin, len(group))
for month in samples:
    samples[month] = pd.concat(samples[month], axis=0)


# Bootstrap analysis to get uncertainty of slopes
bs_table = []
for month in samples:
    strat_samp = samples[month]
    strat_samp = strat_samp.loc[(strat_samp['log_bin'] > 0) & (strat_samp['log_bin'] < 10)]
    mle_results = []
    ls_results = []
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

        beta, a, r, p, err = linregress(np.log(resamp['L']), np.log(resamp['total_deformation']*24*60*60))
        ls_results.append(-beta)
    
    q1_mle, q2_mle = np.quantile(np.array(mle_results), [0.025, 0.975])
    q1_ls, q2_ls = np.quantile(np.array(ls_results), [0.025, 0.975])
    bs_table.append([month, q1_mle, q2_mle, q1_ls, q2_ls])
    
bs_table = pd.DataFrame(bs_table, columns=['month', 'min_beta_mle', 'max_beta_mle', 'min_beta_lsq', 'max_beta_lsq']).set_index('month')
bs_table['beta_mle'] = np.nan
bs_table['beta_lsq'] = np.nan

# Calculation of slopes over stratified sample
for month in samples:
    strat_samp = samples[month]
    strat_samp = strat_samp.loc[(strat_samp['log_bin'] > 0) & (strat_samp['log_bin'] < 10)]
    likelihood_results = pd.Series(np.nan, index=np.linspace(0.01, 1, 200))
    for beta in likelihood_results.index:
        likelihood_results.loc[beta] = normal_log_likelihood(strat_samp['total_deformation']*(60*60*24),
                                                             strat_samp['L'], beta)
    beta = likelihood_results.idxmax()
    bs_table.loc[month, 'beta_mle'] = beta 

    beta, a, r, p, err = linregress(np.log(strat_samp['L']), np.log(strat_samp['total_deformation']*24*60*60))
    bs_table.loc[month, 'beta_lsq'] = -beta
    bs_table.loc[month, 'a_lsq'] = a

###### Plotting #########
fig, axs = pplt.subplots(ncols=3)
for ax, month, monthname in zip(axs, range(4, 7), ['April', 'May', 'June']):
    strat_samp = samples[month]
    strat_samp['log_total_deformation'] = np.log(strat_samp['total_deformation'])
    data = strat_samp.loc[(strat_samp.datetime.dt.month == month) & (strat_samp.no_overlap_sample)]
    data = data.loc[(data.log_bin > 0) & (data.log_bin < 10)]
    ax.scatter(data['L'].values, data['total_deformation'].values*24*60*60, m='.', color='blue3', ms=2, zorder=0, label='')
    data_mean = data.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').mean()
    ax.plot(data_mean['L'].values, data_mean['total_deformation'].values*24*60*60, marker='o', color='k')

    # Uncomment to plot the standard deviation
    # data_stdv = data.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').std()
    # ax.plot(data_mean['L'].values, data_stdv['total_deformation'].values*24*60*60, marker='+', ls='--', color='k')

    
    n = data.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').count()    
    beta_mle = bs_table.loc[month, 'beta_mle']
    scaled_eps = data['total_deformation']*(60*60*24)*data['L']**beta_mle
    mu = np.mean(np.log(scaled_eps))
    sigma = np.std(np.log(scaled_eps))
    mean = np.exp(mu + sigma**2/2)
    stdev = np.sqrt((np.exp(sigma**2)-1)*np.exp(2*mu + sigma**2))
    ax.plot(data_mean['L'].values, (np.exp(mu)*data_mean['L']**(-beta_mle)).values, label='', color='r', marker='^', ms=5, lw=1, zorder=5)
    ax.plot(data_mean['L'].values, (mean*data_mean['L']**(-beta_mle)).values, label='', color='r', marker='.', ms=5, lw=1, zorder=5)
    ax.plot(data_mean['L'].values, (stdev*data_mean['L']**(-beta_mle)).values, label='', ls='--', color='r', marker='+', ms=5, lw=1, zorder=5)


    
    min_beta = bs_table.loc[month, 'min_beta_mle']
    max_beta = bs_table.loc[month, 'max_beta_mle']    
    mle_result = '$\\beta={b} \, ({minb}, {maxb})$'.format(b=np.round(beta_mle, 2),
                                                              minb=np.round(min_beta, 2),
                                                              maxb=np.round(max_beta, 2))
    
    # Can uncomment to show the LSQ results text summary, they are identical
    # beta_lsq = bs_table.loc[month, 'beta_lsq']
    # min_beta = bs_table.loc[month, 'min_beta_lsq']
    # max_beta = bs_table.loc[month, 'max_beta_lsq']   
    # lr_result = '$\\beta={b:.2f} \, ({minb:.2f}, {maxb:.2f})$'.format(b=np.round(beta_lsq, 2),
    #                                                                   minb=np.round(min_beta, 2),
    #                                                                   maxb=np.round(max_beta, 2))

    df_quantiles = pd.DataFrame({q: strat_samp[['log_bin', 'total_deformation']].groupby('log_bin').quantile(q).values.squeeze()
                                 for q in [0.1, 0.25, 0.5, .75, 0.9]},
             index = strat_samp[['log_bin', 'L']].groupby('log_bin').mean().values.squeeze())

    for q, ls in zip([0.1, 0.25, 0.5, .75, 0.9], [':', '--', '-', '--', ':']):
        if q == 0.5:
            m = '^'
        else:
            m = ''
        ax.plot(df_quantiles.index, df_quantiles[q].values*24*60*60, color='blue8', ls=ls, lw=2, m=m)
    # Uncomment to show LSQ results
    # ax.format(lltitle='MLE: ' + mle_result + '\n' + 'LSQ: ' + lr_result, xreverse=False)
    ax.format(lltitle=mle_result)
    ax.format(yscale='log', xscale='log', ylim=(0.9e-3, 1.5), xlabel='Length scale (km)',
              ylabel='Total deformation ($s^{-1}$)', xlim=(9, 150), title=monthname, xreverse=False)

h = [ax.plot([],[], c=c, lw=lw, m=m, ms=ms, ls=ls) for c, lw, m, ms, ls in zip(
                ['blue3', 'k', 'blue8', 'blue8', 'blue8', 'r', 'r'],
                [0, 1, 1, 1, 1, 1, 1],
                ['o', 'o', '^', '', '', '.', '^'],
                [5, 5, 5, 0, 0, 0, 4, 5],
                ['', '-', '-', '--', ':', '-', '-'])]
                 #Obs / Mean / Median / 25-75

# make custom legend
ax.legend(h, ['Observations', 'Mean', 'Median', '25-75%', '10-90%', 'MLE Mean', 'MLE Median'], ncols=1,loc='r')
fig.save('../figures/fig10_deformation_scales.png', dpi=300)
fig.save('../figures/fig10_deformation_scales.pdf', dpi=300)