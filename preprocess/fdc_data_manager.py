"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Data manager for processing the FDC dataset.

To-do:
"""
# standard libraries
import logging as log
import os

# third party libraries
import pandas as pd

class FdcDataManager:
    """
    Class for managing the FDC data.
    """
    def __init__(self, fdc_dir, configparser):
        """
        Class initializer.

        Inputs:
            fdc_dir: (str) Directory containing the FDC data.
            configparser: (ConfigParser) Object related to the
                .ini configuration file for FDC data.
        """
        self.fdc_dir = fdc_dir
        self.configparser = configparser
        self.fdc_data_dic = self._load_data()

    def _load_data(self):
        """
        (Private) Load FDC data according to the configuration file.

        Returns:
            fdc_data_dic: (dict) Dictionary where key is the FDC filename,
                and the value is DataFrame containing its data.
        """
        fdc_data_dic = {}

        for filename in self.configparser.sections():
            filepath = os.path.join(self.fdc_dir, '{}.csv'.format(filename))
            log.info('Loading FDC %s data from \'%s\'...', filename, filepath)

            dtype_dic = {}
            usecols_list = []
            parse_dates_list = []

            for key in self.configparser.options(filename):
                for value in self.configparser.getstr(key, section=filename).split(', '):
                    if key == 'datetime':
                        parse_dates_list.append(value)
                    elif key == 'usecols':
                        usecols_list.append(value)
                    else:
                        dtype_dic[value] = key

            log.debug('dtype: %s', str(dtype_dic))
            log.debug('use columns: %s', str(usecols_list))
            log.debug('datetime columns: %s', str(parse_dates_list))

            fdc_data_dic[filename] = pd.read_csv(
                filepath,
                dtype=dtype_dic,
                parse_dates=parse_dates_list)[usecols_list]

        return fdc_data_dic

    def join_data(self):
        """
        Join the FDC data using 'fdc_id' as the main key.

        Returns:
            pd_joined: (DataFrame) Joined data.
        """
        # join (food, branded_food)
        pd_joined = self.fdc_data_dic['food'].join(
            self.fdc_data_dic['branded_food'].set_index('fdc_id'),
            on='fdc_id')

        # join (joined, survey_fndds_food)
        pd_joined = pd_joined.join(
            self.fdc_data_dic['survey_fndds_food'].set_index('fdc_id'),
            on='fdc_id')

        # join (joined, wweia_food_category)
        pd_joined = pd_joined.join(
            self.fdc_data_dic['wweia_food_category'].set_index('wweia_food_category_code'),
            on='wweia_category_code')

        # join (joined, agricultural_acquisition)
        pd_joined = pd_joined.join(
            self.fdc_data_dic['agricultural_acquisition'].set_index('fdc_id'),
            on='fdc_id')

        # join (joined, food_category)
        pd_joined = pd_joined.join(
            self.fdc_data_dic['food_category'].set_index('id'),
            on='food_category_id',
            rsuffix='_food_category')

        # join (food_attribute, food_attribute_type)
        self.fdc_data_dic['food_attribute'] = self.fdc_data_dic['food_attribute'].fillna('')
        self.fdc_data_dic['food_attribute'] = self.fdc_data_dic['food_attribute'].groupby(
            ['fdc_id', 'food_attribute_type_id'])['value'].agg(', '.join).reset_index()

        pd_food_attribute_joined = self.fdc_data_dic['food_attribute'].join(
            self.fdc_data_dic['food_attribute_type'].set_index('id'),
            on='food_attribute_type_id')

        pd_food_attribute_joined = pd_food_attribute_joined.groupby(
            ['fdc_id']).agg(', '.join).reset_index()

        # join (joined, food_attribute_joined)
        pd_joined = pd_joined.join(
            pd_food_attribute_joined.set_index('fdc_id'),
            on='fdc_id',
            rsuffix='_food_attribute')

        return pd_joined
