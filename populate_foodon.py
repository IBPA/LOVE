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
import sys
from time import time

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
    args = parse_argument()
    configparser = ConfigParser(args.config_file)
    set_logging(configparser.getstr('logfile'))

    # parse FoodOn
    parse_foodon = ParseFoodOn(configparser.getstr('foodon_parse_config'))
    classes_dict = parse_foodon.get_candidate_classes()
    classes_dict_skeleton, candidate_entities = parse_foodon.get_seeded_skeleton(classes_dict)

    # run
    scoring_manager = ScoringManager(
        classes_dict_skeleton,
        candidate_entities,
        configparser.getstr('scoring_config'))

    scoring_manager.run_iteration()


if __name__ == '__main__':
    main()
