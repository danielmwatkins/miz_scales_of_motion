import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
import ultraplot as pplt
import pyproj
import scipy.stats as stats
from scipy.interpolate import interp2d
import sys
import warnings
import xarray as xr

pplt.rc.reso = 'med'
pplt.rc['geo.round'] = False

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

df_ift = pd.read_csv('../data/floe_tracker/ift_floe_trajectories.csv', index_col=0)
df_ift['datetime'] = pd.to_datetime(df_ift['datetime'].values)

# Calculations
# Length scale bins need area adjustment
df_ift['area_adj_km2'] = (np.sqrt(df_ift.area) + 8)**2*.25*.25 # 6 pixel shift minimizes error against manual

edge_bins = np.arange(0, 800, 25)
df_ift['edge_bin'] = np.digitize(df_ift.edge_dist_km, bins=edge_bins)

length_bins = np.arange(0, 50, 2)
df_ift['length_scale'] = df_ift['area_adj_km2']**0.5
df_ift['length_bin'] = np.digitize(df_ift.length_scale, bins=length_bins)

ds_sic = xr.open_dataset('../data/nsidc_cdr_subset.nc').load()
sic_monthly_mean = ds_sic.groupby('time.month').mean(dim='time')

##### Calculate Velocity Mean and Comparison with NSIDC ######
min_x = df_ift.x_stere.min()
max_x = df_ift.x_stere.max()
min_y = df_ift.y_stere.min()
max_y = df_ift.y_stere.max()

x_bins = np.arange(min_x, max_x, 25e3)
y_bins = np.arange(min_y, max_y, 25e3)
xc = 0.5*(x_bins[1:] + x_bins[:-1])
yc = 0.5*(y_bins[1:] + y_bins[:-1])
X, Y = np.meshgrid(xc, yc)

# Select data from 25 km bins for IFT and for NSIDC to compute monthly means
crs0 = pyproj.CRS('WGS84')
crs1 = pyproj.CRS('epsg:3413')
transformer_pstere = pyproj.Transformer.from_crs(crs1, crs_to=crs0, always_xy=True)

lon_grid, lat_grid = transformer_pstere.transform(np.ravel(X), np.ravel(Y))
lon_grid = np.reshape(lon_grid, X.shape)
lat_grid = np.reshape(lat_grid, Y.shape)

u_data = {}
v_data = {}
hist = {}
diffs_mean = {}
diffs_u = {}
diffs_v = {}

for month in [4, 5, 6]:
    u_data[month] = {}
    v_data[month] = {}
    hist[month] = {}
    for label, suffix in zip(['IFT', 'NSIDC'], ['', '_nsidc']):
        sel = (df_ift.datetime.dt.month == month) & (df_ift['u'].notnull())
        x = df_ift.x_stere
        y = df_ift.y_stere
        u = df_ift['u' + suffix] * 100
        v = df_ift['v' + suffix] * 100
        
        hist2d = np.histogram2d(df_ift.loc[sel, :].dropna(subset=['u'], axis=0)['x_stere'], # only drop where IFT is missing
                       df_ift.loc[sel, :].dropna(subset=['v'], axis=0)['y_stere'],
                      bins=[x_bins, y_bins])
        
        df_hist = pd.DataFrame(hist2d[0], index=xc, columns=yc)
        hist[month] = df_hist
        
        u_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
            x[sel], y[sel], values=u[sel], statistic='mean', 
            bins=[x_bins, y_bins])
        v_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
            x[sel], y[sel], values=v[sel], statistic='mean', 
            bins=[x_bins, y_bins])

        # Rotation from Earth coordinates to the north polar stereographic grid for display
        U_nps = u_mean.T * np.sin(np.deg2rad(lon_grid + 45)) + v_mean.T * np.cos(np.deg2rad(lon_grid + 45))
        V_nps = v_mean.T * np.cos(np.deg2rad(lon_grid + 45)) - u_mean.T * np.sin(np.deg2rad(lon_grid + 45))
        u_data[month][label] = pd.DataFrame(U_nps.T, index=xc, columns=yc)
        v_data[month][label] = pd.DataFrame(V_nps.T, index=xc, columns=yc)

    # Calculate the mean of the differences from vector components
    # and calculate the L2 norm of the difference 
    
    diff_u = (df_ift['u'] - df_ift['u_nsidc']) * 100
    diff_v = (df_ift['v'] - df_ift['v_nsidc']) * 100
    diff_norm = np.sqrt(diff_u**2 + diff_v**2)
    dn, xedges, yedges, binnumber = stats.binned_statistic_2d(
                x[sel], y[sel], values=diff_norm[sel], statistic='mean', 
                bins=[x_bins, y_bins])
    diffs_mean[month] = pd.DataFrame(dn, index=xc, columns=yc)
    du, xedges, yedges, binnumber = stats.binned_statistic_2d(
                x[sel], y[sel], values=diff_u[sel], statistic='mean', 
                bins=[x_bins, y_bins])
    dv, xedges, yedges, binnumber = stats.binned_statistic_2d(
                x[sel], y[sel], values=diff_v[sel], statistic='mean', 
                bins=[x_bins, y_bins])
    
    # Rotation from Earth coordinates to the north polar stereographic grid for display
    U_nps = du.T * np.sin(np.deg2rad(lon_grid + 45)) + dv.T * np.cos(np.deg2rad(lon_grid + 45))
    V_nps = dv.T * np.cos(np.deg2rad(lon_grid + 45)) - du.T * np.sin(np.deg2rad(lon_grid + 45))
    diffs_u[month] = pd.DataFrame(U_nps.T, index=xc, columns=yc)
    diffs_v[month] = pd.DataFrame(V_nps.T, index=xc, columns=yc)

    print('Month', month, 'mean difference', diffs_mean[month].mean().mean().round(2))
    
ds_sic = xr.open_dataset('../data/nsidc_cdr_subset.nc').load()
sic_monthly_mean = ds_sic.groupby('time.month').mean(dim='time')
# sic_monthly_mean = sic_monthly_mean.where('cdr_seaice_conc') <= 1

crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
fig, axs = pplt.subplots(height=8, proj='npstere',
                         proj_kw={'lon_0': -45}, ncols=6, nrows=2, share=False)


h = []
for ls in ['--', '-']:
    h.append(axs[0,0].plot([], [], ls=ls, color='k'))
l = ['15%', '85%']


month_names = {4: 'April', 5: 'May', 6: 'June'}
for ax, month in zip(axs, [4, 5, 6]*4):
    ax.set_extent([0.49e6, 1.1e6, -2.05e6, -0.77e6], crs=crs)  
    ax.format(land=True, coast=True, lonlocator=np.arange(-30, 30, 10),
              latmax=85, latlocator=np.arange(55, 85, 5), 
           landzorder=0, landcolor='k', facecolor='w')
    ax.contour(sic_monthly_mean.x, sic_monthly_mean.y, sic_monthly_mean.sel(month=month)['cdr_seaice_conc']*100,
               levels=[15, 85, 85.01], ls=['--', '-', '-'], color='k', transform=crs, zorder=3) # weird bug with ony two levels
    if month == 4:
        ax.legend(h, l, loc='ll', ncols=1, alpha=1, label='SIC') 
    ax.format(title=month_names[month])

xspacing = 2
yspacing = 1

# Plot IFT
for ax, month in zip(axs[0, 0:3], month_names):
    idx_data = hist[month] > 30
    speed = np.sqrt(u_data[month]['IFT'] **2 + v_data[month]['IFT'] **2)
    c1 = ax.pcolor(lon_grid, lat_grid, speed.where(idx_data).T.values, vmin=0, vmax=30,
           transform=ccrs.PlateCarree(), cmap='coolwarm', alpha=0.9)

    ax.quiver(lon_grid[::xspacing, ::yspacing], lat_grid[::xspacing, ::yspacing],
              u_data[month]['IFT'].where(idx_data).T.values[::xspacing, ::yspacing],
              v_data[month]['IFT'].where(idx_data).T.values[::xspacing, ::yspacing],
           transform=ccrs.PlateCarree(), color='k', scale=90, width=1/150)
ax.colorbar(c1, label='IFT Mean Drift (cm/s)', loc='r', shrink=0.9)

# Plot NSIDC
for ax, month in zip(axs[0, 3:6], month_names):
    idx_data = hist[month] > 30
    speed = np.sqrt(u_data[month][label] **2 + v_data[month][label] **2)
    c2 = ax.pcolor(lon_grid, lat_grid, speed.where(idx_data).T.values, vmin=0, vmax=30,
           transform=ccrs.PlateCarree(), cmap='coolwarm', alpha=0.9)

    ax.quiver(lon_grid[::xspacing, ::yspacing], lat_grid[::xspacing, ::yspacing],
              u_data[month]['NSIDC'].where(idx_data).T.values[::xspacing, ::yspacing],
              v_data[month]['NSIDC'].where(idx_data).T.values[::xspacing, ::yspacing],
           transform=ccrs.PlateCarree(), color='k', scale=90, width=1/150)
ax.colorbar(c2, label='NSIDC Mean Drift (cm/s)', loc='r', shrink=0.9)

# Data Counts
for ax, month in zip(axs[1, 0:3], month_names):
    # Plot count of IFT observations
    idx = hist[month] > 0
    c0 = ax.pcolor(lon_grid, lat_grid, hist[month].where(idx).T, vmin=10, vmax=100,
           transform=ccrs.PlateCarree(), cmap='blues')
ax.colorbar(c0, label='IFT Vector Count', loc='r', shrink=0.9)

# Plot diffs
for ax, month in zip(axs[1, 3:6], month_names):
    idx_data = hist[month] > 30
    c3 = ax.pcolor(lon_grid, lat_grid, diffs_mean[month].where(idx_data).T.values, vmin=0, vmax=21,
           transform=ccrs.PlateCarree(), cmap='reds', N=7, alpha=0.9)
    
    ax.quiver(lon_grid[::xspacing, ::yspacing], lat_grid[::xspacing, ::yspacing],
              diffs_u[month].where(idx_data).T.values[::xspacing, ::yspacing],
              diffs_v[month].where(idx_data).T.values[::xspacing, ::yspacing],
           transform=ccrs.PlateCarree(), color='k', scale=90, width=1/150)
ax.colorbar(c3, label='Difference Magnitude (cm/s)', loc='r', shrink=0.9)
axs.format(abc=True)
for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/fig14_mean_drift.{im}'.format(im=imtype), dpi=300)


