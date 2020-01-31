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
    set_logging()
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    # init FDC data manager
    fdc_dm = FdcDataManager(
        configparser.getstr('fdc_data_dir'),
        configparser.getstr('fdc_data_config_filepath'),
        configparser.getstr('fdc_process_config_filepath'))

    # join the FDC data and save it
    joined_filepath = os.path.join(
        configparser.getstr('output_dir'),
        configparser.getstr('joined_fdc_filename'))

    pd_processed = fdc_dm.join(joined_filepath)

    # do additional processing
    pd_processed = fdc_dm.filter(pd_processed)
    pd_processed = fdc_dm.merge_categories(pd_processed)
    pd_processed = fdc_dm.create_source_column(pd_processed)
    pd_processed = fdc_dm.drop_columns(pd_processed)

    # do final processing and save the final file
    processed_filepath = os.path.join(
        configparser.getstr('output_dir'),
        configparser.getstr('final_fdc_filename'))

    log.info('Saving final processed FDC data to \'%s\'...', processed_filepath)
    pd_processed.to_csv(processed_filepath, sep='\t')

    # init FDC preprocess manager
    fpm = FdcPreprocessManager(
        configparser.getstr('fdc_preprocess_config_filepath'))

    pd_processed['description'] = fpm.preprocess_column(pd_processed['description'])
    pd_processed['ingredients'] = fpm.preprocess_column(pd_processed['ingredients'])
    pd_processed['category'] = fpm.preprocess_column(pd_processed['category'])

    filename_output = os.path.join(
        configparser.getstr('output_dir'),
        configparser.getstr('output_filename'))

    log.info('Saving preprocessed FDC data to \'%s\'...', filename_output)
    pd_processed.to_csv(filename_output, sep='\t')


if __name__ == '__main__':
    main()
