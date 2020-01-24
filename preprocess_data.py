"""
Authors:
    Jason Youn -jyoun@ucdavis.edu
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess the FDC data.

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import re
import sys

# third party imports
import pandas as pd

# local imports
from preprocess.fdc_data_manager import FdcDataManager
from preprocess.token_manager import TokenManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging

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
    data_preprocess_dir = configparser.getstr('data_preprocess_dir', 'input')
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

    log.info('Saving filtered FDC data to \'%s\'...', filtered_filepath)
    pd_filtered.to_csv(filtered_filepath, index=False)

    # preprocessing before tokenization
    pd_token = pd_filtered.copy()
    pd_token['fdc_id'] = pd_token['fdc_id'].astype(int)
    pd_token.set_index('fdc_id', inplace=True)

    columns_omit = ['market_class', 'treatment', 'food_category_id', 'data_type', 'wweia_category_code']
    pd_token.drop(columns_omit, axis=1, inplace=True)

    col_name_map = {'branded_food_category': 'branded',
                    'wweia_food_category_description': 'wweia',
                    'description_food_category': 'SR_legacy'}
    pd_token.rename(columns=col_name_map, inplace=True)

    # choose data that has some kind of category information
    pd_token = pd_token[~pd_token['branded'].isnull() | ~pd_token['wweia'].isnull() | ~pd_token['SR_legacy'].isnull()]
    pd_token['description'] = pd_token['description'].str.lower()
    pd_token['ingredients'] = pd_token['ingredients'].str.lower()

    # tokenize
    tm = TokenManager(data_preprocess_dir)

    pd_token = tm.categorize_pd(pd_token)
    pd_token['description'] = pd_token['description'].map(tm.find_replace)
    pd_token['description'] = pd_token['description'].map(tm.remove_numeric)
    pd_token['description'] = pd_token['description'].map(tm.tokenize)
    pd_token['ingredients'] = pd_token['ingredients'].map(tm.clean_ingredient)
    pd_token['ingredients'] = pd_token['ingredients'].map(tm.find_replace)

    # add nutrient information to tokens
    nutrient_dir = '/home/jyoun/Jason/Research/FoodOntology/data/FDC/FoodData_Central_csv_2019-12-17'
    filename_nut = os.path.join(nutrient_dir, 'food_nutrient.csv')
    pd_nut = pd.read_csv(filename_nut, sep=',', usecols=['fdc_id', 'nutrient_id', 'amount'])
    pd_nut['fdc_id'] = pd_nut['fdc_id'].astype(int)

    filename_nutcat = os.path.join(nutrient_dir, 'nutrient.csv')
    pd_nutcat = pd.read_csv(filename_nutcat)
    nutcat_map = pd_nutcat.set_index('id')['name'].to_dict()
    nutcat_ids = [int(x) for x in '1004,1008,1050,1079,1085,1253,1257,1258,1292,1293'.split(',')]
    nutcat_names = [nutcat_map[x] for x in nutcat_ids]

    for nutcat_id, nutcat_name in zip(nutcat_ids, nutcat_names):
        pd_sel = pd_nut[pd_nut['nutrient_id'] == nutcat_id]
        pd_sel[nutcat_name] = pd_sel['amount']
        pd_token = pd_token.join(pd_sel.set_index('fdc_id')[nutcat_name])

    # clean and save tokens
    if 'index' in pd_token.columns:
        del pd_token['index']

    filename_token = os.path.join(output_dir, 'tokenized.csv')
    pd_token.to_csv(filename_token, sep=';')


if __name__ == '__main__':
    main()
