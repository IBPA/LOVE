#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# root dir
root_dir=`pwd`

# update paths in prepare.ini
default_dir="/path/to/project/root/directory"
user_dir="$root_dir"

echo "Updating filepaths in 'prepare.ini'..."
sed -i 's|'$default_dir'|'$user_dir'|g' "$root_dir/config/prepare.ini"

# update paths in preprocess.ini
echo "Updating filepaths in 'preprocess.ini'..."
sed -i 's|'$default_dir'|'$user_dir'|g' "$root_dir/config/preprocess.ini"
