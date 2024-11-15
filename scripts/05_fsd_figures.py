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

#### Low SIC vs High SIC ####


#### All vs Tracked Floes ####
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