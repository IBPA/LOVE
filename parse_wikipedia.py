"""
Authors:
    Jason Youn -jyoun@ucdavis.edu
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Prepare the FDC data.

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import sys

# third party imports
import pandas as pd

# local imports
from preprocess.fdc_data_manager import FdcDataManager
from preprocess.fdc_preprocess_manager import FdcPreprocessManager
from preprocess.wikipedia_manager import WikipediaManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging

# global variables
DEFAULT_CONFIG_FILE = './config/prepare.ini'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Prepare data.')

    parser.add_argument(
        '--config_file',
        default=DEFAULT_CONFIG_FILE,
        help='Path to the .ini configuration file for processing the FDC data.')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # set log, parse args, and read configuration
    set_logging(log_level=log.INFO)
    args = parse_argument()
    configparser = ConfigParser(args.config_file)


    pd_processed = pd.read_csv('./output/preprocessed.txt', sep='\t', index_col='fdc_id')
    pd_processed.fillna('', inplace=True)

    queries = []
    for row in pd_processed['description_preprocessed'].tolist():
        queries.extend(row.split(' '))

    # queries = queries[0:30]

    wm = WikipediaManager(queries)
    wm.get_summary(
        './output/summaries.txt',
        './output/failed.txt')

if __name__ == '__main__':
    main()
