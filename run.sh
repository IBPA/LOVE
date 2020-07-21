#!/usr/bin/env bash

# exit immediately upon error
set -e

# variables
root_dir=`pwd`

# Look up all the words found in the FDC data on WikiPedia.
# Only the summary section of WikiPedia is downloaded.
# The WikiPedia data is also preprocessed.
python3 parse_wikipedia.py

# Train embeddings of the words from WikiPedia.
python3 train_embeddings.py --config_file ./config/word2vec_wiki.ini

# Do ontology population.
python3 populate_foodon.py
