"""Estimate the length scale parameter associated 95% confidence interval based on bootstrap estimates. 
Each year has a different number of data points."""

# Make monthly sample set
rs = 32413
n = 1000
samples = {4: [], 5: [], 6: []}
for (month, log_bin), group in all_results.groupby(['month', 'log_bin']):        
    if len(group) > 1000:
        if log_bin > 0:
            if month in samples:
                samples[month].append(group.sample(n, replace=False, random_state=rs + month + log_bin))

# Build bootstrap analysis of slopes
bs_table = []
for month in samples:
    strat_samp = pd.concat(samples[month], axis=0)
    bs_results = []
    for repeat in range(1000):
        resamp = strat_samp.sample(len(strat_samp), replace=True, random_state=rs + repeat)
        likelihood_results = pd.Series(np.nan, index=np.linspace(0.01, 1, 200))
        for beta in likelihood_results.index:
            likelihood_results.loc[beta] = normal_log_likelihood(resamp['total_deformation']*(60*60*24),
                                                                 resamp['L'], beta)
        bs_results.append(likelihood_results.idxmax())
    del likelihood_results
    q1, q2 = np.quantile(np.array(bs_results), [0.025, 0.975])
    bs_table.append([month, q1, q2])
    
bs_table = pd.DataFrame(bs_table, columns=['month', 'min_beta', 'max_beta']).set_index('month')
bs_table['beta'] = np.nan
for month in samples:
    strat_samp = pd.concat(samples[month], axis=0)
    likelihood_results = pd.Series(np.nan, index=np.linspace(0.01, 1, 200))
    for beta in likelihood_results.index:
        likelihood_results.loc[beta] = normal_log_likelihood(strat_samp['total_deformation']*(60*60*24),
                                                             strat_samp['L'], beta)
    beta = likelihood_results.idxmax()
    bs_table.loc[month, 'beta'] = beta
    

fig, axs = pplt.subplots(ncols=3, nrows=1, spanx=False)
for ax, month, monthname in zip(axs, [4, 5, 6],
                                ['April', 'May', 'June', 'July']):
    beta = bs_table.loc[month, 'beta']
    min_beta = bs_table.loc[month, 'min_beta']
    max_beta = bs_table.loc[month, 'max_beta']    
    all_month = all_results.loc[all_results.month==month]
    ax.scatter(all_month['L'].values, all_month['total_deformation'].values*(60*60*24), marker='.', alpha=0.05, ms=5, color='steelblue')
    ax.scatter(strat_samp['L'].values, strat_samp['total_deformation'].values*(60*60*24), 
              marker='.', alpha=0.5, ms=5, color='k')
        
    for val, ls in zip([beta], ['-']):
        scaled_eps = strat_samp['total_deformation']*(60*60*24)*strat_samp['L']**val
        mu = np.mean(np.log(scaled_eps))
        sigma = np.std(np.log(scaled_eps))
        mean = np.exp(mu + sigma**2/2)
    
        ax.plot([10, 400], [mean*10**(val * -1), mean*400**(val * -1)], color='r', ls=ls)
    
    ax.format(xscale='log', yscale='log', xlim=(8, 500), title=monthname, ylim=(1e-5, 10))
    ax.text(10, 1e-4, '$\\beta={b} \, ({minb}, {maxb})$'.format(b=np.round(beta, 2),
                                                              minb=np.round(min_beta, 2),
                                                              maxb=np.round(max_beta, 2)), color='r')

    h = [ax.plot([],[], m='s', lw=0, ms=5, color='tab:blue'),
         ax.plot([],[], m='s', lw=0, ms=5, color='k')]
ax.legend(h, ['All data', 'Sample'], ncols=1, loc='lr')
axs.format(ylabel='Total deformation (day$^{-1}$', xlabel='Length scale (km)', abc=True)
fig.save('../figures/fig06_deformation_length_scale.jpg', dpi=300)