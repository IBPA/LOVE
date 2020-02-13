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

# local imports
from managers.fdc_data import FdcDataManager
from managers.fdc_preprocess import FdcPreprocessManager
from managers.wikipedia import WikipediaManager
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
    pd_processed = fdc_dm.join(configparser.getstr('joined_fdc_filename'))

    # do additional processing
    pd_processed = fdc_dm.filter(pd_processed)
    pd_processed = fdc_dm.merge_categories(pd_processed)
    pd_processed = fdc_dm.create_source_column(pd_processed)
    pd_processed = fdc_dm.drop_columns(pd_processed)

    # do final processing and save the final file
    log.info('Saving final processed FDC data to \'%s\'...',
        configparser.getstr('final_fdc_filename'))
    pd_processed.to_csv(configparser.getstr('final_fdc_filename'), sep='\t')

    # init FDC preprocess manager
    fpm = FdcPreprocessManager(
        configparser.getstr('fdc_preprocess_config_filepath'))

    # preprocess columns
    pd_processed['concatenated'] = pd_processed[['description', 'ingredients', 'category']].agg(' '.join, axis=1)
    pd_processed['concatenated_preprocessed'] = fpm.preprocess_column(pd_processed['concatenated'])

    # get vocabs and save
    log.info('Saving FDC vocabularies to \'%s\'...',
        configparser.getstr('vocabs_filename'))

    vocabs = fpm.get_vocabs(pd_processed['concatenated_preprocessed'])

    with open(configparser.getstr('vocabs_filename'), 'w') as file:
        for vocab in vocabs:
            file.write('{}\n'.format(vocab))

    # save preprocess final data
    log.info('Saving preprocessed FDC data to \'%s\'...',
        configparser.getstr('preprocessed_filename'))
    pd_processed.to_csv(configparser.getstr('preprocessed_filename'), sep='\t')


if __name__ == '__main__':
    main()
