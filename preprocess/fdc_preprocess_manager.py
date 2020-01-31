"""
Authors:
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess manager for processing the FDC dataset.

To-do:
"""
# standard imports
import logging as log
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Text8Corpus
import gensim.parsing.preprocessing as gpp
from gensim.test.utils import datapath
import numpy as np
import pandas as pd

# local imports
from utils.config_parser import ConfigParser


class FdcPreprocessManager:
    """
    Class for preprocessing the FDC data.
    """

    def __init__(self, config_filepath):
        """
        Class initializer.

        Inputs:
        """
        self.configparser = ConfigParser(config_filepath)

        self.data_preprocess_dir = self.configparser.getstr(
            'data_preprocess_dir', 'directory')

        self.phrase_model_output_dir = self.configparser.getstr(
            'phrase_model_output_dir', 'directory')

    def _build_custom_filter_list(self, which):
        custom_filters = []

        if self.configparser.getbool('lower', which):
            log.debug('Converting to lower cases for %s', which)
            custom_filters.append(lambda x: x.lower())

        if self.configparser.getbool('strip_punctuation', which):
            log.debug('Stripping punctuation for %s', which)
            custom_filters.append(gpp.strip_punctuation)

        if self.configparser.getbool('strip_multiple_whitespaces', which):
            log.debug('Stripping multiple whitespaces for %s', which)
            custom_filters.append(gpp.strip_multiple_whitespaces)

        if self.configparser.getbool('strip_numeric', which):
            log.debug('Stripping numeric for %s', which)
            custom_filters.append(gpp.strip_numeric)

        if self.configparser.getbool('remove_stopwords', which):
            log.debug('Removing stopwords for %s', which)
            custom_filters.append(gpp.remove_stopwords)

        if self.configparser.getbool('strip_short', which):
            minsize = self.configparser.getint('strip_short_minsize', which)
            log.debug('Stripping words shorter than %d for %s', minsize, which)
            custom_filters.append(lambda x: gpp.strip_short(x, minsize=minsize))

        if self.configparser.getbool('stem_text', which):
            log.debug('Stemming text for %s', which)
            custom_filters.append(gpp.stem_text)

        return custom_filters

    def _generate_phrase(self, pd_data, which):
        if self.configparser.getbool('generate_phrase', which):
            log.info('Generating phrases using the %s...', which)
            model = Phrases(pd_data.tolist())

            log.info('Applying phrase model to the %s...', which)
            pd_data = pd_data.apply(
                lambda x: model[x],
                convert_dtype=False)

            model_filepath = os.path.join(
                self.phrase_model_output_dir,
                self.configparser.getstr('phrase_model_filename', which))

            log.info('Saving %s phrase model to \'%s\'...', which, model_filepath)
            model.save(model_filepath)
        else:
            log.info('Skipping phrase generation for %s...', which)

        return pd_data

    def preprocess_column(self, pd_data):
        # preprocess using set of filters
        custom_filters = self._build_custom_filter_list(pd_data.name)

        log.info('Applying preprocess filters to the %s...', pd_data.name)
        pd_data = pd_data.apply(
            lambda x: gpp.preprocess_string(x, custom_filters),
            convert_dtype=False)

        # generate phrase based on the configuration
        pd_data = self._generate_phrase(pd_data, pd_data.name)

        # join the list of words into space delimited string
        pd_data = pd_data.apply(lambda x: ' '.join(x))

        return pd_data
