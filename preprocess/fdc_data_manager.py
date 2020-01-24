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
import numpy as np
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

            dtype = {}
            option_value = self.configparser.get_section_as_dict(filename)

            for key, values in option_value.items():
                if key not in ['datetime', 'usecols']:
                    for value in values:
                        dtype[value.strip()] = key

            datetime = [] if 'datetime' not in option_value else option_value['datetime']
            usecols = [] if 'usecols' not in option_value else option_value['usecols']

            log.debug('dtype: %s', str(dtype))
            log.debug('datetime columns: %s', str(datetime))
            log.debug('using only these columns: %s', str(usecols))

            fdc_data_dic[filename] = pd.read_csv(
                filepath,
                dtype=dtype,
                parse_dates=datetime)[usecols]

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
#        self.fdc_data_dic['food_attribute'] = self.fdc_data_dic['food_attribute'].fillna('')
#        self.fdc_data_dic['food_attribute'] = self.fdc_data_dic['food_attribute'].groupby(
#            ['fdc_id', 'food_attribute_type_id'])['value'].agg(', '.join).reset_index()
#
#        pd_food_attribute_joined = self.fdc_data_dic['food_attribute'].join(
#            self.fdc_data_dic['food_attribute_type'].set_index('id'),
#            on='food_attribute_type_id')
#
#        pd_food_attribute_joined = pd_food_attribute_joined.groupby(
#            ['fdc_id']).agg(', '.join).reset_index()
#
#        # join (joined, food_attribute_joined)
#        pd_joined = pd_joined.join(
#            pd_food_attribute_joined.set_index('fdc_id'),
#            on='fdc_id',
#            rsuffix='_food_attribute')

        return pd_joined

    def filter_data(self, pd_data, column_keyword):
        """
        Filter the rows of joined data that contains
        the keyword(s) located at certain column(s).

        Inputs:
            pd_data: (DataFrame) Joined FDC data.
            column_keyword: (dict) Dictionary where key is the
                column name, and value is list of keywords
                that will be searched in the specified column.
                The value field can also be a regex.
                ex) column_keyword = {
                        'data_type': 'sample_food|market_acquisition',
                        'description': ['bean'],
                    }

        Returns:
            (DataFrame) Filtered data.
        """
        idx = pd.Series(False, index=np.arange(pd_data.shape[0]))

        for column, keyword in column_keyword.items():
            if isinstance(keyword, list):
                keyword = '|'.join(keyword)

            idx |= pd_data[column].str.contains(keyword, case=False)

        return pd_data[idx].reset_index()
