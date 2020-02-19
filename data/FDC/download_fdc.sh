#!/usr/bin/env bash

# exit immediately upon error
set -e

# variables
fdc_base_url="https://fdc.nal.usda.gov/fdc-datasets"
all_foods_csv_filename="FoodData_Central_csv_2019-12-17.zip"
all_foods_dir=${all_foods_csv_filename%.zip}
all_foods_download_url="$fdc_base_url/$all_foods_csv_filename"

# import functions
. ../../utils/utilities.sh

# download FDC all foods dataset, extract
if ! dir_exists_and_is_not_empty $all_foods_dir; then
	echo "Downloading FDC all foods dataset..."
	wget $all_foods_download_url
	unzip $all_foods_csv_filename -d $all_foods_dir
fi
