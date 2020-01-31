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

        data_preprocess_dir = self.configparser.getstr('data_preprocess_dir')



    def preprocess_description(self, pd_description):
        custom_filters = [
            lambda x: x.lower(),
            gpp.strip_punctuation,
            gpp.strip_multiple_whitespaces,
            gpp.strip_numeric,
            gpp.remove_stopwords,
            lambda x: gpp.strip_short(x, minsize=2),
            gpp.stem_text]

        log.info('Applying preprocessing filters to the description...')
        pd_preprocessed = pd_description.apply(
            lambda x: gpp.preprocess_string(x, custom_filters),
            convert_dtype=False)

        log.info('Generating phrases using the description...')
        phrases = Phrases(pd_preprocessed.tolist())

        log.info('Applying phrase model to the description...')
        pd_preprocessed = pd_preprocessed.apply(
            lambda x: phrases[x],
            convert_dtype=False)

        return pd_preprocessed

    def preprocess_ingredient(self, pd_ingredient):
        custom_filters = [
            lambda x: x.lower(),
            gpp.strip_punctuation,
            gpp.strip_multiple_whitespaces,
            gpp.strip_numeric,
            gpp.remove_stopwords,
            lambda x: gpp.strip_short(x, minsize=2),
            gpp.stem_text]

        log.info('Applying preprocessing filters to the ingredient...')
        pd_preprocessed = pd_ingredient.apply(
            lambda row: gpp.preprocess_string(row, custom_filters),
            convert_dtype=False)

        return pd_preprocessed

    def preprocess_category(self, pd_category):
        custom_filters = [
            lambda x: x.lower(),
            gpp.strip_punctuation,
            gpp.strip_multiple_whitespaces,
            gpp.strip_numeric,
            gpp.remove_stopwords,
            lambda x: gpp.strip_short(x, minsize=2),
            gpp.stem_text]

        log.info('Applying preprocessing filters to the category...')
        pd_preprocessed = pd_category.apply(
            lambda row: gpp.preprocess_string(row, custom_filters),
            convert_dtype=False)

        return pd_preprocessed


