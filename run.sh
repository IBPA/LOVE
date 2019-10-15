#!/usr/bin/env bash

# exit immediately upon error
set -e

# variables
root_dir=`pwd`

# download data
cd $root_dir/data/FDC
./download_fdc.sh

# run preprocessor
cd $root_dir
python3 preprocess_data.py
