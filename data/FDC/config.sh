#!/usr/bin/env bash

fdc_base_url="https://fdc.nal.usda.gov/fdc-datasets"
all_foods_csv_filename="FoodData_Central_csv_2019-04-02.zip"
all_foods_dir=${all_foods_csv_filename%.zip}
all_foods_download_url="$fdc_base_url/$all_foods_csv_filename"
