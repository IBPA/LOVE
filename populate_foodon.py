"""
Authors:
    Jason Youn - jyoun@ucdavis.edu
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess the FDC data.

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import sys

# third party imports
import pandas as pd
from gensim.models import KeyedVectors

# local imports
from managers.parse_foodon import ParseFoodOn
from managers.scoring import ScoringManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging

# global variables
DEFAULT_CONFIG_FILE = './config/populate_foodon.ini'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Populate FoodON.')

    parser.add_argument(
        '--config_file',
        default=DEFAULT_CONFIG_FILE,
        help='Path to the .ini configuration file.')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # set log, parse args, and read configuration
    set_logging()
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    parse_foodon = ParseFoodOn(configparser.getstr('foodon_parse_config'))
    all_classes = parse_foodon.get_classes()
    candidate_classes, candidate_entities = parse_foodon.get_seeded_skeleton(all_classes)

    keyed_vectors = KeyedVectors.load_word2vec_format('./data/model/fdc_wiki_embeddings.txt')
    # keyed_vectors.init_sims(replace=True)
    # keyed_vectors.save('./output/temp_embeddings.txt')
    # sys.exit()

    # keyed_vectors = KeyedVectors.load('./output/temp_embeddings.txt', mmap='r')

    candidate_entities = list(set(candidate_entities))
    candidate_entities.sort()

    scoring_manager = ScoringManager(
        keyed_vectors,
        candidate_classes,
        candidate_entities,
        configparser.getstr('preprocess_config'),
        configparser.getstr('scoring_config'))

    scoring_manager.run_iteration()


if __name__ == '__main__':
    main()
