"""
Authors:
    Jason Youn - jyoun@ucdavis.edu
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess manager for the FDC dataset.

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
            config_filepath: (str) Configuration filepath.
        """
        self.configparser = ConfigParser(config_filepath)

        self.data_preprocess_dir = self.configparser.getstr(
            'data_preprocess_dir', 'directory')
        self.phrase_model_output_dir = self.configparser.getstr(
            'phrase_model_output_dir', 'directory')
        self.output_dir = self.configparser.getstr(
            'output_dir', 'directory')

    def _build_custom_filter_list(self, which):
        """
        (Private) Build list of filters based on the configuration file
        that will be applied by gpp.preprocess_string().

        Inputs:
            which: (str) Column name ('description' | 'ingredients' | 'category').

        Returns:
            custom_filters: (list) List of functions.
        """
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
        """
        (Private) Generate phrase using the gensim Phrase detection module.

        Inputs:
            pd_data: (pd.Series) Data which will be used to generate phase.
            which: (str) Column name ('description' | 'ingredients' | 'category').

        Returns:
            pd_data: (pd.Series) Input data but using phrases.
        """
        if self.configparser.getbool('generate_phrase', which):
            log.info('Generating phrases using the %s...', which)

            # this is our training data
            sentences = pd_data.tolist()

            # detect phrases using the configuration
            model = Phrases(
                sentences,
                min_count=self.configparser.getint('min_count', which),
                threshold=self.configparser.getfloat('threshold', which),
                max_vocab_size=self.configparser.getint('max_vocab_size', which),
                progress_per=self.configparser.getint('progress_per', which),
                scoring=self.configparser.getstr('scoring', which))

            # apply trained model to generate phrase
            log.info('Applying phrase model to the %s...', which)
            pd_data = pd_data.apply(
                lambda x: model[x],
                convert_dtype=False)

            # save phrase model
            model_filepath = os.path.join(
                self.phrase_model_output_dir,
                self.configparser.getstr('phrase_model_filename', which))

            log.info('Saving %s phrase model to \'%s\'...', which, model_filepath)
            model.save(model_filepath)

            # dump phrase and its score as text
            phrase_score_list = []
            for phrase, score in model.export_phrases(sentences):
                phrase_score_list.append([phrase.decode('utf-8'), score])

            pd_phrase_score = pd.DataFrame(phrase_score_list, columns=['phrase', 'score'])
            pd_phrase_score.drop_duplicates(subset='phrase', inplace=True)

            export_filepath = os.path.join(
                self.output_dir,
                self.configparser.getstr('phrase_dump_filename', which))

            log.info('Dumping %s phrases to \'%s\'...', which, export_filepath)
            pd_phrase_score.to_csv(export_filepath, sep='\t', index=False)
        else:
            log.info('Skipping phrase generation for %s...', which)

        return pd_data

    def preprocess_column(self, pd_data):
        """
        Preprocess specified column.

        Inputs:
            pd_data: (pd.Series) Input data to preprocess.

        Returns:
            pd_data: (pd.Series) Preprocess data.
        """
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
