"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Data manager for processing the FDC dataset downloaded from the website.

To-do:
"""
# standard libraries
import logging as log
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party libraries
import pandas as pd

# local imports
from utils.config_parser import ConfigParser


class FdcDataManager:
    """
    Class for managing the FDC data.
    """
    def __init__(self, fdc_dir, data_config_filepath, process_config_filepath):
        """
        Class initializer.

        Inputs:
            fdc_dir: (str) Directory containing the downloaded FDC data.
            data_config_filepath: (str) Filepath of the configuration file
                containing instructions on how to load the downloaded FDC data.
            process_config_filepath: (str) Filepath of the configuration file
                containing instructions on how to process FDC data.
        """
        self.fdc_dir = fdc_dir
        self.data_configparser = ConfigParser(data_config_filepath)
        self.process_configparser = ConfigParser(process_config_filepath)
        self.synonym_filename ='synonym.csv'

        # load FDC data into a dictionary
        self.fdc_data_dic = self._load_data()

        synonym_filepath = '/Users/tarininaravane/Documents/GitHub/FoodOntology/data/FDC/synonym.csv'
        pd_synonym = pd.read_csv(synonym_filepath, sep=',')
        self.dict_word_synonym = pd_synonym.set_index('Word')['Synonym'].to_dict()

    def _load_data(self):
        """
        (Private) Load FDC data according to the configuration file.

        Returns:
            fdc_data_dic: (dict) Dictionary where key is the FDC filename,
                and the value is DataFrame containing the loaded data.
        """
        fdc_data_dic = {}
        filenames = self.data_configparser.sections()

        # make sure we are going to load files
        assert len(filenames) != 0

        for filename in filenames:
            filepath = os.path.join(self.fdc_dir, '{}.csv'.format(filename))
            log.info('Loading FDC %s data from \'%s\'...', filename, filepath)

            dtype = {}
            option_value = self.data_configparser.get_section_as_dict(filename)

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

    def join(self, save_to=None):
        """
        Join the FDC data using 'fdc_id' as the main key.

        Inputs:
            save_to: (str, optional) Path to save the data to.

        Returns:
            pd_joined: (pd.DataFrame) Joined data.
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

        # # join (food_attribute, food_attribute_type)
        # self.fdc_data_dic['food_attribute'] = self.fdc_data_dic['food_attribute'].fillna('')
        # self.fdc_data_dic['food_attribute'] = self.fdc_data_dic['food_attribute'].groupby(
        #     ['fdc_id', 'food_attribute_type_id'])['value'].agg(', '.join).reset_index()

        # pd_food_attribute_joined = self.fdc_data_dic['food_attribute'].join(
        #     self.fdc_data_dic['food_attribute_type'].set_index('id'),
        #     on='food_attribute_type_id')

        # pd_food_attribute_joined = pd_food_attribute_joined.groupby(
        #     ['fdc_id']).agg(', '.join).reset_index()

        # # join (joined, food_attribute_joined)
        # pd_joined = pd_joined.join(
        #     pd_food_attribute_joined.set_index('fdc_id'),
        #     on='fdc_id',
        #     rsuffix='_food_attribute')

        # set fdc_id as index
        pd_joined['fdc_id'] = pd_joined['fdc_id'].astype(int)
        pd_joined.set_index('fdc_id', inplace=True)

        # set all nan with empty string
        pd_joined.fillna('', inplace=True)

        if save_to:
            log.info('Saving joined FDC data to \'%s\'...', save_to)
            pd_joined.to_csv(save_to, sep='\t')

        return pd_joined

    def filter(self, pd_data, save_to=None):
        """
        Filter the rows of joined data that contains the keyword(s) located at certain column(s).

        Inputs:
            pd_data: (pd.DataFrame) Joined FDC data.
            save_to: (str, optional) Path to save the data to.

        Returns:
            pd_filtered: (pd.DataFrame) Filtered data.
        """
        column_keyword = self.process_configparser.get_section_as_dict('filter_fdc_data')

        # check if dictionary is empty
        if not bool(column_keyword):
            log.info('Nothing to filter. Skipping data filtering...')
            return pd_data

        idx = pd.Series(False, index=pd_data.index)

        for column, keyword in column_keyword.items():
            if isinstance(keyword, list):
                keyword = '|'.join(keyword)

            idx |= pd_data[column].str.contains(keyword, case=False)

        pd_filtered = pd_data[idx]

        if save_to:
            log.info('Saving filtered FDC data to \'%s\'...', save_to)
            pd_filtered.to_csv(save_to, sep='\t')

        return pd_filtered

    def merge_categories(self, pd_data, save_to=None):
        """
        Merge categories into one column.

        Inputs:
            pd_data: (pd.DataFrame) Input data.
            save_to: (str, optional) Path to save the data to.

        Returns:
            pd_merged: (pd.DataFrame) Categories merged.
        """
        pd_merged = pd_data.copy()

        if set(['from', 'to']).issubset(self.process_configparser.options('category_merge')):
            merge_list = self.process_configparser.getstr('from', 'category_merge').split(', ')
            merge_to = self.process_configparser.getstr('to', 'category_merge')

            log.info('Merging categories %s into \'%s\'', str(merge_list), merge_to)
            pd_merged[merge_to] = pd_merged[merge_list].apply(lambda x: ' '.join(x), axis=1)
        else:
            log.info('Not merging categories')

        if save_to:
            log.info('Saving FDC data with merged columns to \'%s\'...', save_to)
            pd_merged.to_csv(save_to, sep='\t')

        return pd_merged

    def create_source_column(self, pd_data, save_to=None):
        """
        Create source column based on the category information.

        Inputs:
            pd_data: (pd.DataFrame) Input data.
            save_to: (str, optional) Path to save the data to.

        Returns:
            pd_output: (pd.DataFrame) Data with source column added.
        """
        pd_output = pd_data.copy()

        source_dict = self.process_configparser.get_section_as_dict(
            'create_source_column', value_delim=None)

        # check if dictionary is empty
        if not bool(source_dict):
            log.info('Not creating source column...')
            return pd_output

        # add and fill source column based on the configuration
        pd_output['source'] = ''

        for column, keyword in source_dict.items():
            idx = ~(pd_output[column] == '')
            pd_output['source'][idx] = keyword

        if save_to:
            log.info('Saving filtered FDC data to \'%s\'...', save_to)
            pd_output.to_csv(save_to, sep='\t')

        return pd_output

    def drop_columns(self, pd_data, save_to=None):
        """
        Drop columns specified in the configuration file.

        Inputs:
            pd_data: (pd.DataFrame) Input data.
            save_to: (str, optional) Path to save the data to.

        Returns:
            pd_dropped: (pd.DataFrame) Data with some columns dropped.
        """
        pd_dropped = pd_data.copy()

        # drop any columns specified by the config file
        if 'drop_columns' in self.process_configparser.options('drop_options'):
            drop_columns = self.process_configparser.getstr(
                'drop_columns', 'drop_options').split(', ')

            log.info('Dropping columns %s', str(drop_columns))
            pd_dropped.drop(drop_columns, axis=1, inplace=True)
        else:
            log.info('No columns to drop')

        if save_to:
            log.info('Saving FDC data with dropped columns to \'%s\'...', save_to)
            pd_dropped.to_csv(save_to, sep='\t')

        return pd_dropped

    def remove_synonyms(self,pd_data):
        pd_data['description'] = pd_data['description'].str.lower()
        #pd_data['ingredients'] = pd_data['ingredients'].str.lower()
        pd_data['description'] = pd_data['description'].map(self.replace_word_synonym)
        #pd_data['ingredients'] = pd_data['ingredients'].map(replace_word_synonym)
        return pd_data

   # Make change for lower case
    def replace_word_synonym(self,x):
        keys = self.dict_word_synonym.keys()
        for key in keys:
            x = x.replace(key, self.dict_word_synonym[key])
        return x
        




