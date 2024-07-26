"""Copy the latest version of the processed Fram Strait imagery from the archive.
TBD: set up to use the Brown Digital Repository API."""

from shutil import copyfile

dataloc = '/Volumes/Research/ENG_Wilhelmus_Shared/group/IFT_fram_strait_dataset/'
saveloc = '../data/floe_tracker/'
for year in range(2003, 2021):
    copyfile(src=dataloc + 'fram_strait-{y}/ift_clean_floe_properties_{y}.csv'.format(y=year),
          dst=saveloc + 'ift_floe_property_tables/clean/ift_clean_floe_properties_{y}.csv'.format(y=year))

    copyfile(src=dataloc + 'fram_strait-{y}/ift_raw_floe_properties_{y}.csv'.format(y=year),
          dst=saveloc + 'ift_floe_property_tables/raw/ift_raw_floe_properties_{y}.csv'.format(y=year))

    copyfile(src=dataloc + 'fram_strait-{y}/ift_interp_floe_trajectories_{y}.csv'.format(y=year),
          dst=saveloc + 'ift_floe_trajectories/ift_interp_floe_trajectories_{y}.csv'.format(y=year))
