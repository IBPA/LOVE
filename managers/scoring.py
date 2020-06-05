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
from time import time
import multiprocessing
import itertools
import math
import random

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party libraries
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import textdistance
import nltk

# local imports
from fdc_preprocess import FdcPreprocessManager
from utils.config_parser import ConfigParser
from utils.utilities import file_exists, save_pkl, load_pkl


class ScoringManager:
    """
    Class for finding scores.
    """
    def __init__(
            self,
            candidate_classes_info,
            candidate_entities,
            scoring_config):
        """
        Class initializer.
        """
        # config parser
        if isinstance(scoring_config, str):
            scoring_config = ConfigParser(scoring_config)

        # save arguments
        self.candidate_classes_info = candidate_classes_info
        self.candidate_entities = candidate_entities

        # parse config file
        self.alpha = scoring_config.getfloat('alpha')
        self.num_mapping_per_iteration = scoring_config.getint('num_mapping_per_iteration')
        self.initial_siblings_scores = scoring_config.getstr('initial_siblings_scores')
        self.initial_parents_scores = scoring_config.getstr('initial_parents_scores')
        self.pairs_filepath = scoring_config.getstr('pairs_filepath')
        self.populated_filepath = scoring_config.getstr('populated_filepath')
        self.preprocess_config_filepath = scoring_config.getstr('preprocess_config')
        self.similarity_method = scoring_config.getstr('similarity_method')

        log.debug('alpha: %f', self.alpha)
        log.debug('num_mapping_per_iteration: %d', self.num_mapping_per_iteration)
        log.debug('initial_siblings_scores: %s', self.initial_siblings_scores)
        log.debug('initial_parents_scores: %s', self.initial_parents_scores)
        log.debug('pairs_filepath: %s', self.pairs_filepath)
        log.debug('populated_filepath: %s', self.populated_filepath)
        log.debug('similarity_method: %s', self.similarity_method)

        # preprocess manager
        self.fpm = FdcPreprocessManager(self.preprocess_config_filepath)

        # number of candidate classes & entities
        self.num_candidate_classes = len(self.candidate_classes_info)
        self.num_candidate_entities = len(self.candidate_entities)

        log.debug('Number of candidate classes: %d', self.num_candidate_classes)
        log.debug('Number of candidate entities: %d', self.num_candidate_entities)

        # extract the seeded entities to make complete list of entities
        seed_entities = self._unpack_sublist([x[1] for _, x in self.candidate_classes_info.items()])
        self.all_entity_labels = list(set(self.candidate_entities + seed_entities))

        # all labels of candidate class
        self.candidate_classes_label = list(self.candidate_classes_info.keys())

        # complete list of class labels
        other_classes = self._unpack_sublist(
            [x[0] for _, x in self.candidate_classes_info.items()],
            depth=2)
        self.all_class_labels = list(set(self.candidate_classes_label + other_classes))

        # calculate embedding lookup table for class / entity labels
        if 'we_' in self.similarity_method:
            self.keyed_vectors = KeyedVectors.load_word2vec_format(
                scoring_config.getstr('word_embeddings'))
            # self.keyed_vectors.save('./output/glove_wiki_embeddings.bin')
            # self.keyed_vectors = KeyedVectors.load('./output/glove_wiki_embeddings.bin')

            self.pd_class_label_embeddings = self._calculate_label_embeddings(self.all_class_labels)
            self.pd_entity_label_embeddings = self._calculate_label_embeddings(self.all_entity_labels)

            # save_pkl(self.pd_class_label_embeddings, './output/pd_class_label_embeddings.pkl')
            # save_pkl(self.pd_entity_label_embeddings, './output/pd_entity_label_embeddings.pkl')
            # sys.exit()

            # self.pd_class_label_embeddings = load_pkl('./output/pd_class_label_embeddings.pkl')
            # self.pd_entity_label_embeddings = load_pkl('./output/pd_entity_label_embeddings.pkl')

        # do initial calculation of the scores
        self.pd_siblings_scores, self.pd_parents_scores = self._calculate_initial_scores()

    @staticmethod
    def _unpack_sublist(input_list, depth=1):
        for i in range(depth):
            input_list = [item for sublist in input_list for item in sublist]

        return list(set(input_list))

    @staticmethod
    def _cosine_similarity(array1, array2):
        with np.errstate(all='ignore'):
            similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

        if np.isnan(similarity):
            similarity = -1

        return similarity

    @staticmethod
    def _euclidean_similarity(array1, array2):
        distance = np.linalg.norm(array1-array2)
        if distance == 0:
            return np.inf
        else:
            return (1 / distance)

    def _caculate_embeddings(self, label):
        label_embedding = 0
        num_found_words = 0

        for word, pos in nltk.pos_tag(label.split(' ')):
            # if word in ['food', 'product']:
            #     continue

            try:
                word_embedding = self.keyed_vectors.wv.word_vec(word)
            except KeyError:
                pass
            else:
                if pos == 'NN':
                    multiplier = 1.15
                else:
                    multiplier = 1

                label_embedding += (multiplier * word_embedding)
                num_found_words += 1

        if num_found_words == 0:
            return np.zeros(300)
        else:
            return label_embedding / num_found_words

    def _calculate_label_embeddings(self, index_list):
        pd_label_embeddings = pd.DataFrame(
            index=index_list,
            columns=['preprocessed', 'vector'])

        pd_label_embeddings['preprocessed'] = self.fpm.preprocess_column(
            pd_label_embeddings.index.to_series(),
            load_model=True)

        # some preprocessed columns are empty due to lemmatiazation, fill it up with original
        empty_index = (pd_label_embeddings['preprocessed'] == '')
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.index.to_series()[empty_index]
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.loc[empty_index, 'preprocessed'].apply(lambda x: x.lower())

        pd_label_embeddings['vector'] = pd_label_embeddings['preprocessed'].apply(
            self._caculate_embeddings)

        return pd_label_embeddings

    def _calculate_siblings_score(self, pair):
        class_label, entity_label = pair[0], pair[1]
        siblings = self.candidate_classes_info[class_label][1]

        if self.similarity_method == 'we_cos':
            num_nonzero_siblings = 0
            siblings_embedding = 0

            for sibling in siblings:
                sibling_embedding = self.pd_entity_label_embeddings.loc[sibling, 'vector']

                if np.count_nonzero(sibling_embedding):
                    siblings_embedding += sibling_embedding
                    num_nonzero_siblings += 1

            if num_nonzero_siblings == 0:
                return 0

            siblings_embedding /= num_nonzero_siblings
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

            score = self._cosine_similarity(siblings_embedding, entity_embeddings)
        elif self.similarity_method == 'we_euc':
            num_nonzero_siblings = 0
            siblings_embedding = 0

            for sibling in siblings:
                sibling_embedding = self.pd_entity_label_embeddings.loc[sibling, 'vector']

                if np.count_nonzero(sibling_embedding):
                    siblings_embedding += sibling_embedding
                    num_nonzero_siblings += 1

            if num_nonzero_siblings == 0:
                return 0

            siblings_embedding /= num_nonzero_siblings
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

            score = self._euclidean_similarity(siblings_embedding, entity_embeddings)
        elif self.similarity_method == 'hamming':
            score = 0
            for sibling in siblings:
                score += textdistance.hamming.normalized_similarity(sibling, entity_label)
            score /= len(siblings)
        elif self.similarity_method == 'jaccard':
            score = 0
            for sibling in siblings:
                score += textdistance.jaccard.normalized_similarity(sibling, entity_label)
            score /= len(siblings)
        elif self.similarity_method == 'lcsseq':
            score = 0
            for sibling in siblings:
                score += textdistance.lcsseq.normalized_similarity(sibling, entity_label)
            score /= len(siblings)
        elif self.similarity_method == 'random':
            score = random.uniform(0, 1)
        else:
            raise ValueError('Invalid similarity method: %s', self.similarity_method)

        return score

    def _calculate_parents_score(self, pair):
        class_label, entity_label = pair[0], pair[1]

        if self.similarity_method == 'we_cos':
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']
            class_embeddings = self.pd_class_label_embeddings.loc[class_label, 'vector']
            score = self._cosine_similarity(class_embeddings, entity_embeddings)
        elif self.similarity_method == 'we_euc':
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']
            class_embeddings = self.pd_class_label_embeddings.loc[class_label, 'vector']
            score = self._euclidean_similarity(class_embeddings, entity_embeddings)
        elif self.similarity_method == 'hamming':
            score = textdistance.hamming.normalized_similarity(class_label, entity_label)
        elif self.similarity_method == 'jaccard':
            score = textdistance.jaccard.normalized_similarity(class_label, entity_label)
        elif self.similarity_method == 'lcsseq':
            score = textdistance.lcsseq.normalized_similarity(class_label, entity_label)
        elif self.similarity_method == 'random':
            score = random.uniform(0, 1)
        else:
            raise ValueError('Invalid similarity method: %s', self.similarity_method)

        return score

    def _calculate_initial_scores(self):
        # calculate siblings score
        if file_exists(self.initial_siblings_scores):
            log.info('Pre-calculated siblings scores found.')
            pd_siblings_scores = pd.read_csv(self.initial_siblings_scores, index_col=0)
        else:
            entity_class_pairs = list(itertools.product(
                self.candidate_classes_label,
                self.candidate_entities))

            log.info('Calculating siblings score...')

            t1 = time()
            with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
                results = p.map(self._calculate_siblings_score, entity_class_pairs)
            t2 = time()

            log.info('Elapsed time for calculating siblings score: %.2f minutes', (t2-t1)/60)

            results = np.array(results).reshape(
                self.num_candidate_classes, self.num_candidate_entities)

            pd_siblings_scores = pd.DataFrame(
                results,
                index=self.candidate_classes_label,
                columns=self.candidate_entities)

            pd_siblings_scores.to_csv(self.initial_siblings_scores)

        # calculate parents score
        if file_exists(self.initial_parents_scores):
            log.info('Pre-calculated parents scores found.')
            pd_parents_scores = pd.read_csv(self.initial_parents_scores, index_col=0)
        else:
            entity_class_pairs = list(itertools.product(
                self.candidate_classes_label,
                self.candidate_entities))

            log.info('Calculating parents score...')

            t1 = time()
            with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
                results = p.map(self._calculate_parents_score, entity_class_pairs)
            t2 = time()

            log.info('Elapsed time for calculating parents score: %.2f minutes', (t2-t1)/60)

            results = np.array(results).reshape(
                self.num_candidate_classes, self.num_candidate_entities)

            pd_parents_scores = pd.DataFrame(
                results,
                index=self.candidate_classes_label,
                columns=self.candidate_entities)

            pd_parents_scores.to_csv(self.initial_parents_scores)

        return pd_siblings_scores, pd_parents_scores

    def run_iteration(self):
        if file_exists(self.pairs_filepath) and file_exists(self.populated_filepath):
            log.info('Pre-calculated iterations found.')
            iteration_pairs = load_pkl(self.pairs_filepath)
            iteration_populated_dict = load_pkl(self.populated_filepath)
            return iteration_pairs, iteration_populated_dict

        num_iterations = math.floor(self.num_candidate_entities / self.num_mapping_per_iteration)
        iteration_pairs = {}
        iteration_populated_dict = {}

        iteration = 0
        while len(self.candidate_entities) > 0:
            log.info('Updating scores. Iteration: %d/%d', iteration, num_iterations)
            t1 = time()

            # calculate score
            pd_scores = self.alpha*self.pd_siblings_scores + (1-self.alpha)*self.pd_parents_scores

            # find top N unique entities with highest score
            num_scores = pd_scores.shape[0] * pd_scores.shape[1]
            pd_top_scores = pd_scores.stack().nlargest(num_scores).reset_index()
            pd_top_scores.columns = ['candidate class', 'candidate entity', 'score']
            pd_top_scores.drop_duplicates(subset='candidate entity', inplace=True)

            log.debug('Top scores: \n%s', str(pd_top_scores.head()))

            top_n_scores = list(zip(
                pd_top_scores['candidate class'], pd_top_scores['candidate entity']))
            top_n_scores = top_n_scores[0:self.num_mapping_per_iteration]

            # populate skeleton using selected entity
            for (candidate_class, candidate_entity) in top_n_scores:
                self.candidate_classes_info[candidate_class][1].append(candidate_entity)

            # save progress
            iteration_pairs[iteration] = top_n_scores.copy()
            iteration_populated_dict[iteration] = self.candidate_classes_info.copy()

            if len(self.candidate_entities) <= self.num_mapping_per_iteration:
                break

            classes_to_update = list(set([x[0] for x in top_n_scores]))
            entities_to_remove = list(set([x[1] for x in top_n_scores]))

            # remove selected entities from candidate entities and scores
            self.candidate_entities = list(set(self.candidate_entities) - set(entities_to_remove))
            self.pd_siblings_scores = self.pd_siblings_scores.drop(
                labels=entities_to_remove, axis=1)
            self.pd_parents_scores = self.pd_parents_scores.drop(
                labels=entities_to_remove, axis=1)

            # if alpha is 0, no need to update siblings score
            if self.alpha == 0.0:
                log.info('Skipping siblings score update since alpha is 0.')
            else:
                # update siblings score
                entity_class_pairs = list(itertools.product(
                    classes_to_update,
                    self.candidate_entities))

                results = []
                for pair in entity_class_pairs:
                    results.append(self._calculate_siblings_score(pair))

                results = np.array(results).reshape(
                    len(classes_to_update), len(self.candidate_entities))

                pd_siblings_to_update = pd.DataFrame(
                    results,
                    index=classes_to_update,
                    columns=self.candidate_entities)

                self.pd_siblings_scores.update(pd_siblings_to_update)

            t2 = time()
            log.info('Elapsed time for updating scores: %.2f minutes', (t2-t1)/60)

            iteration += 1

        save_pkl(iteration_pairs, self.pairs_filepath)
        save_pkl(iteration_populated_dict, self.populated_filepath)

        return iteration_pairs, iteration_populated_dict
