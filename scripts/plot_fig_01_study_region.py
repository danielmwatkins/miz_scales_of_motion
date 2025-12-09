import ultraplot as pplt
import pandas as pd
from shapely.geometry import shape
import fiona
import cartopy.crs as ccrs
import warnings
import numpy as np

warnings.simplefilter('ignore')
pplt.rc['cartopy.circular'] = False
pplt.rc['reso'] = 'med'

bounds = {'Study Region': [247326, 1115678, -635759, -2089839],
          'Fig. 2 (a-f)': [0.6e6, 1.1e6, -1.5e6, -2e6],
          'Fig. 3': [0.75e6, 0.95e6, -1.6e6, -1.8e6]} # Update with new figure

crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
fig, axs = pplt.subplots(width=4, proj='npstere',
                         proj_kw={'lon_0': -45}, share=False)
axs.format(land=True, lonlocator=np.arange(-100, 30, 10), latmax=85, latlocator=np.arange(55, 85, 5), latlabels=True, lonlabels=True)
axs.set_extent([-0.5e6, 1.7e6, -3.15e6, -0.5e6], crs=crs)  

for c, b, ls in zip([c['color'] for c in pplt.Cycle("Dark2", 4)], bounds, ['-', '-', '-', '--']):
    left, right, top, bottom = bounds[b]
    axs.plot([left, right, right, left, left],
             [top, top, bottom, bottom, top], label='', transform=crs, ls=ls, lw=2, zorder=50, c=c)
    axs.hist([], histtype='step', color=c, lw=2, ls=ls, label=b)

axs.plot([],[], color='blue4', label='April Extent')
axs.plot([], [], color='red4', label='September Extent')
axs.legend(alpha=1, ncols=1, loc='lr', lw=1)

# Open the shapefile
for year in range(2003, 2021):
    for color, month, z in zip(['blue4', 'red4'], ['04', '09'], [0, 30]):
        with fiona.open("../data/nsidc_sea_ice_extent/extent_N_{y}{m}_polyline_v3.0/extent_N_{y}{m}_polyline_v3.0.shp".format(y=year, m=month)) as shapefile:
            for record in shapefile:
                geometry = shape(record['geometry'])
    
        for x in list(geometry.geoms):
            x, y = x.xy
            axs.plot(pd.Series(x).rolling(3, center=True).mean(),
                     pd.Series(y).rolling(3, center=True).mean(),
                     transform=crs, color=color, lw=2, zorder=z, alpha=0.5)

for imtype in ['png', 'pdf']:
    fig.save("../figures/{im}/fig01_study_region.{im}".format(im=imtype), dpi=300)

