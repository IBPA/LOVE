#!/usr/bin/env bash

# exit immediately upon error
set -e

# import config & functions
. config.sh
. ../../utils/utilities.sh

# download FDC all foods dataset, extract, and remove .zip file
if ! dir_exists_and_is_not_empty $all_foods_dir; then
	echo "FDC all foods dataset does not exist."
	echo "Downloading FDC all foods dataset..."
	wget $all_foods_download_url
	unzip $all_foods_csv_filename -d $all_foods_dir
	rm $all_foods_csv_filename
fi
