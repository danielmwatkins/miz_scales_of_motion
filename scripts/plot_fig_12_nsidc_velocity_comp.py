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

ds_asi = xr.open_dataset('../data/asi_sic_merged.nc').load()

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
        
        hist2d = np.histogram2d(df_ift.loc[sel, :].dropna(subset=['u', 'u_nsidc'], axis=0)['x_stere'],
                       df_ift.loc[sel, :].dropna(subset=['v', 'v_nsidc'], axis=0)['y_stere'],
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
    
crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
fig, axs = pplt.subplots(width=8, proj='npstere',
                         proj_kw={'lon_0': -45}, ncols=3, nrows=2, share=False)
# 
for ax in axs:
    ax.set_extent([0.5e6, 0.97e6, -2.05e6, -0.77e6], crs=crs)  
    ax.format(land=True, coast=True, 
           landzorder=0, landcolor='k', facecolor='w')
from matplotlib.patches import Rectangle

for ax in axs[:, 0]:
    left = 0.5e6
    top = -0.77e6
    left_pad = 50e3
    top_pad = 0.13e6
    ax.add_patch(Rectangle([left, top-150e3], 200e3, 150e3, facecolor='w', edgecolor='k', transform=crs, zorder=10))
    ax.plot([left + left_pad, left + left_pad + 100e3], [top - top_pad, top - top_pad],
            lw=4, color='k',transform=crs, zorder=11)
    ax.text(left + left_pad + 30e3, top - 0.1e6, '100 km', color='k', zorder=11, transform=crs)

    ax.quiver(left + left_pad, top - 0.065e6, 20, 0,
              lw=4, color='k', scale=90, width=1/100, transform=crs, zorder=11)
    ax.text(left + left_pad + 30e3, top - 0.04e6, '10 cm/s', color='k', zorder=11, transform=crs)

for col, month in zip([0, 1, 2], [4, 5, 6]):

    idx_data = hist[month] > 30
    
    # Plot count of IFT observations
    c0 = axs[0, col].pcolor(lon_grid, lat_grid, hist[month].where(idx_data).T.values, vmin=0, vmax=100,
           transform=ccrs.PlateCarree(), cmap='blues', extend='max')
    
    axs[0, col].quiver(lon_grid, lat_grid, u_data[month]['IFT'].where(idx_data).T.values, v_data[month]['IFT'].where(idx_data).T.values,
               transform=ccrs.PlateCarree(), color='r', scale=90, width=1/150, label='IFT')
    axs[0, col].quiver(lon_grid, lat_grid, u_data[month]['NSIDC'].where(idx_data).T.values, v_data[month]['NSIDC'].where(idx_data).T.values,
               transform=ccrs.PlateCarree(), color='k', scale=90, width=1/100, label='NSIDC')
    
    axs[0, 0].legend(loc='ll', ncols=1, alpha=1, lw=2)
    
    c1 = axs[1, col].pcolor(lon_grid, lat_grid, diffs_mean[month].where(idx_data).T.values, vmin=0, vmax=20,
           transform=ccrs.PlateCarree(), cmap='reds', extend='max', N=7)
    axs[1, col].quiver(lon_grid, lat_grid, diffs_u[month].where(idx_data).T.values, diffs_v[month].where(idx_data).T.values,
               transform=ccrs.PlateCarree(), color='k', scale=90, width=1/100)

    monthly_mean = ds_asi.sel(time=ds_asi.time.dt.month == month).mean(dim='time')
    axs[0, col].contour(monthly_mean.x, monthly_mean.y, monthly_mean['sic'], levels=[1, 15, 85], ls=[':', '--', '-'], color='k', transform=crs)
    axs[1, col].contour(monthly_mean.x, monthly_mean.y, monthly_mean['sic'], levels=[1, 15, 85], ls=[':', '--', '-'], color='k', transform=crs)
    
        
axs[0, col].colorbar(c0, loc='r', shrink=0.85, label='Count', labelsize=11)
axs[1, col].colorbar(c1, loc='r', shrink=0.85, label='Magnitude of Difference (cm/s)', labelsize=11)
axs.format(leftlabels = ['Mean Drift','Difference'],
           toplabels=['April', 'May', 'June'], fontsize=12, abc=True)
h = []
for ls in [':', '--', '-']:
    h.append(axs[0,0].plot([], [], ls=ls, color='k'))
l = ['1%', '15%', '85%']
axs[0, 2].legend(h, l, loc='lr', ncols=1, alpha=1, label='SIC') 
axs[1, 2].legend(h, l, loc='lr', ncols=1, alpha=1, label='SIC') 

for imtype in ['png', 'pdf']:
    fig.save('../figures/{im}/fig12_mean_drift.{im}'.format(im=imtype), dpi=300)



