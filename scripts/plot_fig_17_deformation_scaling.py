"""Plot deformation scaling results."""
import pandas as pd
import ultraplot as pplt
import numpy as np
import scipy.stats as stats

##### Load deformation data #######
df = pd.read_csv('../data/deformation/sampled_results.csv', index_col=0, parse_dates=['datetime'])
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['relative_error'] = df['uncertainty_total'] / df['total_deformation']
df = df.loc[df.log_bin.between(1, 8)]

n_init = len(df)
df = df.loc[df.relative_error < 0.5].copy()

n_post = len(df)
print('Filtering by relative error reduced number of samples from ', n_init, 'to', n_post)
###### Calculating beta #######
# Stratified sample by log bin
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
            # weight this by the number of observations per day within each bin
    else:
        pass

for month in samples:
    samples[month] = pd.concat(samples[month], axis=0)

bs_table = pd.read_csv('../data/deformation/scaling_estimates.csv', index_col='month')

###### Plotting #########
fig, axs = pplt.subplots(ncols=3, sharex=False)
for ax, month, monthname in zip(axs, range(4, 7), ['April', 'May', 'June']):
    strat_samp = samples[month].loc[samples[month].log_bin.between(1, 5)]
    # strat_samp['log_total_deformation'] = np.log(strat_samp['total_deformation'])
    data = strat_samp.loc[(strat_samp.datetime.dt.month == month) & (strat_samp.no_overlap_sample)]
    data = data.loc[(data.log_bin > 0) & (data.log_bin < 10)]
    ax.scatter(data['L'].values, data['total_deformation'].values*24*60*60, m='.', color='blue3', ms=2, zorder=0, label='')
    
    data_mean = data.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').mean()
    data_std = data.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').std()
    # compute the standard error of the mean
    n = 400
    dist = stats.t(n-1)
    critical_value = dist.ppf(0.975)
    SE = data.groupby('log_bin').apply(lambda x: stats.sem(x['total_deformation']))
    ax.errorbar(
        data_mean['L'].values, data_mean['total_deformation'].values*24*60*60,
        yerr=critical_value * SE.values*24*60*60,
        marker='.', color='k', lw=0, elinewidth=1, capsize=2)    


    n = data.loc[:, ['L', 'total_deformation', 'log_bin']].groupby('log_bin').count()    
    beta_mle = bs_table.loc[month, 'beta_mle']
    scaled_eps = data['total_deformation']*(60*60*24)*data['L']**beta_mle
    mu = np.mean(np.log(scaled_eps))
    sigma = np.std(np.log(scaled_eps))
    mean = np.exp(mu + sigma**2/2)
    stdev = np.sqrt((np.exp(sigma**2)-1)*np.exp(2*mu + sigma**2))
    
    # Log-normal estimate of the median
    L = data_mean['L'].values
    ax.plot(L, np.exp(mu)*L**(-beta_mle), label='', color='r', marker='^', ms=5, lw=1, zorder=5)

    # Log-normal estimate of the mean
    ax.plot(L, mean*L**(-beta_mle), label='', color='r', marker='.', ms=5, lw=1, zorder=5)

    # LSQ slope
    beta_lsq = bs_table.loc[month, 'beta_lsq']
    int_lsq = bs_table.loc[month, 'a_lsq'] # already has had exponent applied
    ax.plot(L, int_lsq*L**(-beta_lsq), label='', color='k', ls='--')

    df_quantiles = pd.DataFrame({q: strat_samp[['log_bin', 'total_deformation']].groupby('log_bin').quantile(q).values.squeeze()
                                 for q in [0.1, 0.25, 0.5, .75, 0.9]},
                                index = strat_samp[['log_bin', 'L']].groupby('log_bin').mean().values.squeeze())

    for q, ls in zip([0.1, 0.25, 0.5, .75, 0.9], [':', '--', '-', '--', ':']):
        if q == 0.5:
            m = '^'
        else:
            m = ''
        ax.plot(df_quantiles.index, df_quantiles[q].values*24*60*60, color='blue8', ls=ls, lw=2, m=m)

    min_beta = bs_table.loc[month, 'min_beta_mle']
    max_beta = bs_table.loc[month, 'max_beta_mle']    
    mle_result = '$\\beta_{m}={b} \, ({minb}, {maxb})$'.format(b=np.round(beta_mle, 2),
                                                              minb=np.round(min_beta, 2),
                                                              maxb=np.round(max_beta, 2),
                                                               m="{MLE}")
    beta_lsq = bs_table.loc[month, 'beta_lsq']
    min_beta = bs_table.loc[month, 'min_beta_lsq']
    max_beta = bs_table.loc[month, 'max_beta_lsq']    
    lsq_result = '$\\beta_{m}={b} \, ({minb}, {maxb})$'.format(b=np.round(beta_lsq, 2),
                                                              minb=np.round(min_beta, 2),
                                                              maxb=np.round(max_beta, 2),
                                                               m="{LSQ}")

    
    # add lsq result text
    ax.format(lltitle=mle_result + '\n' + lsq_result)
    ax.format(yscale='log', xscale='log', ylim=(0.005, 1.5), xlabel='Length scale (km)',
              ylabel='Total deformation (day$^{-1}$)', xlim=(14, 80), title=monthname, xreverse=False)

    h = [ax.plot([],[], c=c, lw=lw, m=m, ms=ms, ls=ls) for c, lw, m, ms, ls in zip(
                    ['blue3', 'k', 'k', 'blue8', 'blue8', 'blue8', 'r', 'r'],
                    [0, 1, 1, 1, 1, 1, 1, 1],
                    ['o', 'o', '', '^', '', '', 'o', '^'],
                    [5, 5, 0, 5, 0, 0, 5, 5],
                    ['', '-', '--', '-', '--', ':', '-', '-'])]

    
    ax.format(yscale='log', xscale='log', xlocator=[20, 30, 40, 50, 60, 70])

# add custom legend
ax.legend(h, ['Observations', 'Mean', 'LSQ Fit', 'Median', '25-75%', '10-90%', 'MLE Mean', 'MLE Median'], ncols=1,loc='r')
axs.format(abc=True)
for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/fig17_deformation_scales.{im}'.format(im=imtype), dpi=300)
fig.save('../figures/fig17_deformation_scales.pdf', dpi=300)