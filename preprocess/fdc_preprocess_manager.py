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
        pd_preprocessed = pd_description.fillna('')
        custom_filters = [
            lambda x: x.lower(),
            gpp.strip_punctuation,
            gpp.strip_multiple_whitespaces,
            gpp.strip_numeric,
            gpp.remove_stopwords,
            gpp.stem_text]

        pd_preprocessed = pd_preprocessed.apply(
            lambda x: gpp.preprocess_string(x, custom_filters),
            convert_dtype=False)

        return pd_preprocessed

    def preprocess_ingredient(self, pd_ingredient):
        pd_preprocessed = pd_ingredient.fillna('')

        custom_filters = [
            gpp.strip_punctuation,
            gpp.strip_multiple_whitespaces,
            gpp.strip_numeric,
            gpp.remove_stopwords,
            gpp.stem_text]

        pd_preprocessed = pd_preprocessed.apply(
            lambda row: gpp.preprocess_string(row, custom_filters),
            convert_dtype=False)

        return pd_preprocessed

