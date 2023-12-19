# scales of motion in the summer marginal ice zone
We examine the spatial and temporal scales of sea ice motion at floe scale using floe trajectories derived from optical satellite imagery. 
The analysis focuses on observations from the East Greenland Sea from 2003-2020 during the sunlit months (April-August). 

# structure
The repository contains folders for data, scripts, figures, and Jupyter notebooks. Github limits the size of data, so we cannot save the 
satellite imagery here, however the processed floe data should be small enough to work. In general: if github lets you upload it, place the 
data in the data folder and give it a descriptive name. I prefer lower case names with no spaces but I won't fight you if you have different 
preferences. Code to reproduce each step of the analysis and to reproduce each figure should go into the scripts folder. Where possible, use 
relative pathnames so that the code in scripts reads data from the data folder. If the data is too large, place it in a directory outside 
the Git repository and provide instructions on how to obtain it. Analysis code and plotting code should be separated, so that the figures can 
be re-generated without additional computation time. This also lets us make minor changes to figures easily.

# python environment
List the key libraries here. In the final version, we'll attempt to make a yml file that allows users to recreate the correct Conda or 
micromamba environment.  
`proplot`  
`xarray`  
`pandas`  
`numpy`  

# scripts
`01_extract_ift_data.py` This script reads that matlab output Rosalinda generated, then produces CSV files with the time stamps, floe 
properties, and floe ids. (TBD: currently inside a notebook, needs to placed into a script) 
`02_interpolate_ift_data.py` Makes a best estimate of the IFT floe locations at exactly 12 UTC each day to enable velocity estimation. Also compiles floe properties and rotation rates. Adds a qc flag based on speed z-score (must be less than 6), circularity (must be greater than 0.6), total path length, and a speed threshold (maximum speed must be larger than 1 pixel/day).
`03_add_nsidc_info_ift.py` Interpolates NSIDC ice motion and climate data record sea ice concentration to floe locations, and calculates the 
distance to the sea ice edge. TBD: add ERA5 wind data as well. Requires external datasets.
`04_finding_polygons.py` Searches all possible combinations of three sea ice floes, and makes a list of all the combinations with minimum 
interior angle greater than 20 degrees. The polygon data is large - 1.15 GB - so needs to be created locally rather than shared on github.  
`05_calculate_deformation.py` Calculates strain rates.  


# data
`floe_tracker` folder contains `ift_with_era5.csv` which has the interpolated tracked floes with all the floe properties with wind speeds from ERA5 added in. `ift_with_nsidc.csv` is produced first; `ift_with_era5.csv` contains the same data but with the addition of the winds. The subfolder 
`interpolated` has the files for each year as produced by script 2. The subfolder `parsed` has the full IFT data including non-tracked floes 
produced by script 1. 
| --- | --- | --- |
| Column name | definition | units |
'year' | Year (YYYY) |  |
'datetime' | Date and time (YYYY-mm-dd HH:MM) | |
'floe_id' | Floe ID | |
'x_stere' | meters |
'y_stere' | meters |
'longitude' | decimal degrees |
'latitude' | decimal degrees |
'area'
'perimeter'
'major_axis'
'minor_axis'
'zeta',
'zeta_est',
'u' | meters per second
'v' | meters per second
'bearing'
'speed'
'circularity',
'qc_flag',
'sea_ice_concentration', 
'edge_dist',
'u_nsidc', 
'v_nsidc', 
'u5D_nsidc',
'v5D_nsidc',
'u15D_nsidc',
'v15D_nsidc',
'u31D_nsidc',
'v31D_nsidc',
'u_along', 
'v_along',
'u_across',
'v_across',
'U_fluctuating',
'U_along',
'u_wind',
'v_wind',
'wind_speed'

# figures
Figures should be print-ready for Annals of Glaciology. They should be eps, tif, or pdf if possible. (Some figures need to be compressed into 
jpg first and then converted, such as scatterplots with huge numbers of points). The figure width should be 86 mm for single column figures 
and 178 mm for two-column figures. Fontsize should be 9. Font should be Optima, or if that is not available, Arial. 


