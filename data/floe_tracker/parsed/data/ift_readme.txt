Data comes from Rosalinda’s hard drive
UCR/research/ice_tracker/server/single_disp/data

Time info is from SOIT, merged with the information from single_disp/input/sat_YYYY.xslx


Folders:
all_floes = contents of the props.mat file for each year. I’ve added columns with the floe_id for floes that appear in the tracked floes, as well as SOIT image date times, 
stereographic coordinates (NSIDC), and longitude and latitude.

tracked_floes = similar to other folder, but only with data that was tracked over at least two images. Contains an attempt to match up the theta values. Theta measures the 
rotation between pass idx and pass idx+1 of a single satellite. There’s a column with the initial floe id, and a column with ID’s based on rows in the x_fixed.mat file. 






Numerous steps along the way that I’ve had to address:
- Longer trajectories (x_fixed.mat) were to be aligned with the original trajectories (x3.mat) so we can track when trajectories were merged
- The 2020 data was run on corrupted images, so I had to add a step to stretch the image back to the correct size. It had been shorted approximately 60 km in the 
stereographic y direction.
- The rotation data do not perfectly match the sizes of the position data arrays. I looked at both the “THETA_aqua_before.mat” and “THETA_aqua” files, which are in the same 
folder, and used the “_before” files because the other file has way more columns some years, and appears to frequently have more data points than are available in the floe 
positions. Looks like there’s missing data for aqua for 2016, where there’s only 107 columns in the matrix instead of 170. The DATA_THETA files range in number of columns 
from 164 to 307 and it’s unclear how to line them up. I’m looking for ways to verify the rotation data so if anyone has ideas I’m all ears. Table below shows what I’m talking 
about with the issue.




Number of columns in the data matrices for, from left to right
THETA_aqua_before.mat
THETA_terra_before.mat
THETA_aqua.mat
DATA_THETA.mat
Subset of days for Aqua (not a matrix)
x3.mat

2003 166 167 166 167 173 346
2004 170 170 172 172 171 343
2005 164 164 164 164 170 340
2006 168 167 168 167 173 346
2007 166 166 166 166 173 346
2008 170 169 170 169 172 344
2009 172 172 307 307 173 346
2010 171 170 184 184 173 346
2011 172 172 172 172 173 346
2012 157 156 157 156 167 334
2013 172 172 172 172 173 346
2014 164 164 164 164 173 346
2015 170 170 170 170 171 343
2016 107 170 170 170 172 344
2017 150 172 172 172 173 346
2018 164 164 286 286 165 330
2019 172 154 172 172 172 345

