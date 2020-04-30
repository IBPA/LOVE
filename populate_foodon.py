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
import sys

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
    classes_info = parse_foodon.get_classes()
    classes_info_skeleton, candidate_entities = parse_foodon.get_seeded_skeleton(classes_info)

    candidate_entities = list(set(candidate_entities))
    candidate_entities.sort()

    scoring_manager = ScoringManager(
        classes_info_skeleton,
        candidate_entities,
        configparser.getstr('preprocess_config'),
        configparser.getstr('scoring_config'))

    iteration_dict = scoring_manager.run_iteration()


if __name__ == '__main__':
    main()
