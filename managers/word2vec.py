"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Lookup queries on wikipedia and save them.

To-do:
"""
# standard imports
import ast
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pylab as plt

# local imports
from utils.config_parser import ConfigParser


class EpochCallback(CallbackAny2Vec):
    """
    Callback to log information about training.
    """
    def __init__(self):
        self.epoch = 0
        self.loss = {}

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        if self.epoch == 0:
            actual_loss = loss
        else:
            actual_loss = loss - self.previous_loss

        log.debug('Loss after epoch %d: %d', self.epoch, actual_loss)

        self.loss[self.epoch] = actual_loss
        self.previous_loss = loss
        self.epoch += 1

    # def on_train_begin(self, model):
    #     print("Beginning training")

    # def on_train_end(self, model):
    #     print("Ending training")

class Word2VecManager():
    """
    """
    def __init__(self, config_file):
        """
        Class initializer.

        Inputs:
        """
        self.configparser = ConfigParser(config_file)
        self.epoch_callback = EpochCallback()
        self.model = None

    def train(self, sentences, pretrained=None):
        self.model = Word2Vec(
            size=self.configparser.getint('size'),
            window=self.configparser.getint('window'),
            min_count=self.configparser.getint('min_count'),
            workers=self.configparser.getint('workers'),
            callbacks=[self.epoch_callback])

        log.info('Building vocabularies...')
        self.model.build_vocab(sentences)
        total_examples = self.model.corpus_count

        if pretrained:
            original_vocabs = self.model.wv.vocab.keys()
            pretrained_vocabs = KeyedVectors.load_word2vec_format(pretrained).vocab.keys()
            common_vocabs = list(set(original_vocabs) & set(pretrained_vocabs))
            log.info('Intersecting %d common vocabularies for transfer learning', len(common_vocabs))

            if self.configparser.getbool('pre_train_update_vocab'):
                log.info('Updating vocabularies using vocabs from pre-trained data')
                self.model.build_vocab([list(pretrained_vocabs)], update=True, min_count=1)

            self.model.intersect_word2vec_format(pretrained, lockf=1.0)

        self.model.train(
            sentences,
            total_examples=total_examples,
            epochs=self.configparser.getint('epochs'),
            compute_loss=True)

    def save_model(self, filepath):
        assert self.model is not None

        log.info('Saving model to %s...', filepath)
        self.model.save(filepath)

    def save_vectors(self, filepath):
        assert self.model is not None

        log.info('Saving word embeddings to %s...', filepath)
        self.model.wv.save_word2vec_format(filepath)

    def save_loss(self, filepath):
        assert self.model is not None

        # sorted by key, return a list of tuples
        lists = sorted(self.epoch_callback.loss.items())

        # unpack a list of pairs into two tuples
        x, y = zip(*lists)

        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(filepath)
