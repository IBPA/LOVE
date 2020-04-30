"""
Authors:
    Jason Youn - jyoun@ucdavis.edu
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess the FDC data.

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import sys

# third party imports
import pandas as pd
import gensim

# local imports
from managers.word2vec import Word2VecManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging

# global variables
DEFAULT_CONFIG_FILE = './config/word2vec_fdc.ini'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train word2vec.')

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
    set_logging()
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    # load data to train with
    sentence_column = configparser.getstr('sentence_column')

    pd_data = pd.read_csv(
        configparser.getstr('input_filepath'),
        sep='\t')

    pd_data.fillna('', inplace=True)
    pd_data = pd_data[pd_data[sentence_column] != '']

    # use specified column as sentences
    sentences = pd_data[sentence_column].tolist()
    sentences = [sentence.split() for sentence in sentences]

    # init word2vec manager
    w2vm = Word2VecManager(args.config_file)

    # start training and load pre-training data if prompted
    if configparser.getbool('pre_train'):
        pretrained = configparser.getstr('pre_trained_vectors')
    else:
        pretrained = None

    w2vm.train(sentences, pretrained=pretrained)

    # save word embeddings and model
    w2vm.save_model(configparser.getstr('model_saveto'))
    w2vm.save_vectors(configparser.getstr('vectors_saveto'))
    w2vm.save_loss(configparser.getstr('loss_saveto'))


if __name__ == '__main__':
    main()
