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
import ast
import logging as log
import os
import sys

# third party imports
import pandas as pd

# local imports
from managers.fdc_data import FdcDataManager
from managers.fdc_preprocess import FdcPreprocessManager
from managers.wikipedia import WikipediaManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging

# global variables
DEFAULT_CONFIG_FILE = './config/wikipedia.ini'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Prepare the WikiPedia data.')

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
    set_logging(log_level=log.INFO)
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    # read preprocessed FDC data
    pd_processed = pd.read_csv(
        configparser.getstr('input_filepath'),
        sep='\t',
        index_col='fdc_id')
    pd_processed.fillna('', inplace=True)

    # build vocabulary list
    vocabs = []
    for row in pd_processed['concatenated_preprocessed'].tolist():
        vocabs.extend(row.split(' '))
    vocabs = list(set(vocabs))

    # get summaries of the wikipedia entry
    wm = WikipediaManager(configparser.getstr('stem_lookup_filepath'))

    # check if we're gonna reuse the previous results
    if configparser.getbool('reuse_previous'):
        prev_summary = configparser.getstr('prev_summaries_filepath')
        prev_failed = configparser.getstr('prev_failed_filepath')
    else:
        prev_summary = None
        prev_failed = None

    pd_summary, pd_failed = wm.get_summary(
        vocabs,
        configparser.getint('num_try'),
        prev_summary=prev_summary,
        prev_failed=prev_failed)

    # save results
    log.info('Saving successfully pulled wiki summaries to %s',
        configparser.getstr('summaries_filepath'))

    pd_summary.to_csv(configparser.getstr('summaries_filepath'),
        sep='\t',
        index=False)

    log.info('Saving failed wiki queries to %s',
        configparser.getstr('failed_filepath'))

    pd_failed.to_csv(configparser.getstr('failed_filepath'),
        sep='\t',
        index=False)

    # apply same preprocessing done to the FDC data
    fpm = FdcPreprocessManager(
        configparser.getstr('preprocess_config'))

    # preprocess columns
    pd_summary['summary_preprocessed'] = fpm.preprocess_column(
        pd_summary['summary'],
        load_model=True)

    output_filepath = configparser.getstr('preprocessed_output')

    log.info('Saving preprocessed wikipedia data to %s...', output_filepath)
    pd_summary.to_csv(output_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
