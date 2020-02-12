#!/bin/bash

# Update paths in the configuration file from general path
# to a local machine specific path. This script must be ran
# in the root project directory.
# ./update_paths.sh: general -> user specific
# ./update_paths.sh revert: user specific -> general

# exit immediately if a command exits with a non-zero status
set -e

# parse argument
if [ -z "$1" ]; then
	echo "Setting mode to default."
	mode="default"
else
	mode=$1
fi

# root dir
root_dir=`pwd`

if [ "$mode" = "default" ]; then # general path -> user specific path
	echo "Updating paths according to the local machine..."

	from="/path/to/project/root/directory"
	to="$root_dir"
elif [ "$mode" = "revert" ]; then # user specific path -> general path
	echo "Reverting path to original..."

	from="$root_dir"
	to="/path/to/project/root/directory"
else # invalid option
	echo "Invalid input '$mode'!"
	exit
fi

for filename in "$root_dir/config"/*.ini; do
	echo "Updating filepaths in '$filename'..."
	sed -i 's|'$from'|'$to'|g' $filename
done
