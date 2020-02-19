#!/usr/bin/env bash

# exit immediately upon error
set -e

# variables
glove_base_url="http://nlp.stanford.edu/data"
glove_6b_filename="glove.6B.zip"
glove_6b_dir=${glove_6b_filename%.zip}
glove_6b_download_url="$glove_base_url/$glove_6b_filename"

# import functions
. ../../utils/utilities.sh

# download GloVe 6B dataset, extract, and remove .zip file
if ! dir_exists_and_is_not_empty $glove_6b_dir; then
	# echo "Downloading FDC all foods dataset..."
	# wget $glove_6b_download_url
	unzip $glove_6b_filename -d $glove_6b_dir
	rm $glove_6b_filename
fi

# convert GloVe format to Word2Vec format
for filename in $glove_6b_dir/*.txt; do
	output_name="${filename%.txt}.word2vec.txt"
	echo "Converting word embedding format from GloVe to Word2Vec for $filename..."
	python3 -m gensim.scripts.glove2word2vec --input $filename --output $output_name
done
