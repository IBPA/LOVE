"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Preprocess the data.

To-do:
"""
# standard imports
import argparse
import logging as log
import os

# third party imports
# import pandas as pd

# local imports
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from preprocess.fdc_data_manager import FdcDataManager

# global variables
DEFAULT_CONFIG_FILE = './config/preprocess.ini'

def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess data.')

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
    set_logging()
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    # config
    output_dir = configparser.getstr('output_dir', 'output')

    # init FDC data manager
    fdc_dm = FdcDataManager(
        configparser.getstr('fdc_data_dir', 'input'),
        ConfigParser(configparser.getstr('fdc_config_filepath', 'input')))

    # join the FDC data and save it
    pd_joined = fdc_dm.join_data()

    joined_filepath = os.path.join(
        output_dir,
        configparser.getstr('joined_fdc_filename', 'output'))

    log.info('Saving joined FDC data to \'%s\'...', joined_filepath)
    pd_joined.to_csv(joined_filepath, index=False)

    # filter the data according to keywords and save
    column_keyword = configparser.get_section_as_dict('filter_fdc_data')
    pd_filtered = fdc_dm.filter_data(pd_joined, column_keyword)

    filtered_filepath = os.path.join(
        output_dir,
        configparser.getstr('filtered_fdc_filename', 'output'))

    pd_filtered.to_csv(filtered_filepath, index=False)

if __name__ == '__main__':
    main()
