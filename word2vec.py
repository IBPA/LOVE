"""
Authors:
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess the FDC data.

To-do:
"""
# standard imports
import os
import random
import numpy as np

# third party imports
import pandas as pd
import gensim

from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def main():
    """
    Main function.
    """

    # settings
    preprocess_dir = '/home/jyoun/Jason/Research/FoodOntology/output'
    filename_token = os.path.join(preprocess_dir, 'preprocessed.txt')

    model_dir = '/home/jyoun/Jason/Research/FoodOntology/data/word2vec'
    filename_model = os.path.join(model_dir, 'modelfile')
    flag_retrain = True

    # model hyperparameters
    hidden_layer_size = 24
    window_size = 100
    min_count = 10
    workers_num = 4
    epochs = 100

    # load tokens
    pd_token = pd.read_csv(filename_token, sep='\t', index_col='fdc_id')
    pd_token['token'] = pd_token['description_preprocessed']
    pd_token = pd_token[pd_token['token'] != '']

    sentences = []
    for tokens in pd_token['token']:
        sentences.append(tokens.split())

    epoch_logger = EpochLogger()

    # train word2vec
    if os.path.exists(filename_model) and not flag_retrain:
        word2vec = gensim.models.Word2Vec.load(filename_model)
    else:
        word2vec = gensim.models.Word2Vec(
            size=hidden_layer_size,
            window=window_size,
            min_count=min_count,
            workers=workers_num,
            callbacks=[epoch_logger],
            iter=1)

        word2vec.build_vocab(sentences)

        word2vec.train(
            sentences,
            total_examples=len(sentences),
            epochs=epochs)

        word2vec.save(filename_model)


if __name__ == '__main__':
    main()
