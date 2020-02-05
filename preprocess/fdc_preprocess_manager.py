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
import gensim.utils as gensim_utils
from gensim.models.phrases import Phrases, Phraser
import gensim.parsing.preprocessing as gpp
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

        self.data_custom_stopwords_dir = self.configparser.getstr(
            'data_custom_stopwords_dir', 'directory')
        self.phrase_model_output_dir = self.configparser.getstr(
            'phrase_model_output_dir', 'directory')
        self.output_dir = self.configparser.getstr(
            'output_dir', 'directory')

    def _generate_custom_stopwords(self, which):
        """
        (Private) Generate custom stopwords by adding or removing
        user specified stopwords to the gensim's default stopwords.

        Inputs:
            which: (str) Column name ('description' | 'ingredients' | 'category').

        Returns:
            (frozenset) New updated stopwords.
        """
        my_stopwords = list(gpp.STOPWORDS)

        # stopwords to add
        to_add_filename = os.path.join(
            self.data_custom_stopwords_dir,
            self.configparser.getstr('stopwords_to_add', which))

        with open(to_add_filename, 'r') as file:
            to_add_list = file.read().splitlines()

        if len(to_add_list) > 0:
            log.info('Adding custom stopwords %s for %s', to_add_list, which)
        else:
            log.info('Not adding any custom stopwords for %s', which)

        # stopwords to remove
        to_remove_filename = os.path.join(
            self.data_custom_stopwords_dir,
            self.configparser.getstr('stopwords_to_remove', which))

        with open(to_remove_filename, 'r') as file:
            to_remove_list = file.read().splitlines()

        if len(to_remove_list) > 0:
            log.info('Removing stopwords %s for %s', to_remove_list, which)
        else:
            log.info('Not removing any custom stopwords for %s', which)

        # add and remove stopwords
        my_stopwords.extend(to_add_list)
        my_stopwords = [x for x in my_stopwords if x not in to_remove_list]

        return frozenset(my_stopwords)

    def _custom_remove_stopwords(self, s, stopwords):
        """
        (Private) Custom remove stopwords function.

        Inputs:
            s: (str) String to process.
            stopwords: (frozenset) Custom stopwords.

        Returns:
            (str) Preprocessed string with stopwords removed.
        """
        s = gensim_utils.to_unicode(s)
        return " ".join(w for w in s.split() if w not in stopwords)

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
            stopwords = self._generate_custom_stopwords(which)
            custom_filters.append(lambda x: self._custom_remove_stopwords(x, stopwords))

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
