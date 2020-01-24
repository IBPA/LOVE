#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# root dir
root_dir=`pwd`

# update paths in preprocess.ini
change_from="/path/to/project/root/directory"
change_to="$root_dir"

echo "Updating filepaths in 'preprocess.ini'..."
sed -i 's|'$change_from'|'$change_to'|g' "$root_dir/config/preprocess.ini"
