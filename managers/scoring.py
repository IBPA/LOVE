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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party libraries
import numpy as np
import pandas as pd

# local imports
from fdc_preprocess import FdcPreprocessManager
from utils.config_parser import ConfigParser
from utils.utilities import file_exists


class ScoringManager:
    """
    Class for finding scores.
    """
    def __init__(
            self,
            keyed_vectors,
            candidate_classes_info,
            candidate_entities,
            preprocess_config,
            scoring_config):
        """
        Class initializer.
        """
        self.keyed_vectors = keyed_vectors
        self.candidate_classes_info = candidate_classes_info
        self.candidate_entities = candidate_entities
        self.fpm = FdcPreprocessManager(preprocess_config)

        # parse config file
        scoring_config = ConfigParser(scoring_config)
        self.alpha = scoring_config.getfloat('alpha')
        self.num_mapping_per_iteration = scoring_config.getint('num_mapping_per_iteration')
        self.initial_siblings_scores = scoring_config.getstr('initial_siblings_scores')
        self.initial_parents_scores = scoring_config.getstr('initial_parents_scores')

        # number of candidate classes & entities
        self.num_candidate_classes = len(self.candidate_classes_info)
        self.num_candidate_entities = len(self.candidate_entities)

        log.info('Number of candidate classes: %d', self.num_candidate_classes)
        log.info('Number of candidate entities: %d', self.num_candidate_entities)

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
        self.pd_class_label_embeddings = self._calculate_label_embeddings(self.all_class_labels)
        self.pd_entity_label_embeddings = self._calculate_label_embeddings(self.all_entity_labels)

        # do initial calculation of the scores
        self.pd_siblings_scores, self.pd_parents_scores = self._calculate_initial_scores()

    @staticmethod
    def _unpack_sublist(input_list, depth=1):
        for i in range(depth):
            input_list = [item for sublist in input_list for item in sublist]

        return list(set(input_list))

    @staticmethod
    def _calculate_cosine_similarity(array1, array2):
        with np.errstate(all='ignore'):
            similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

        if np.isnan(similarity):
            similarity = 0

        return (similarity + 1.0) / 2

    def _caculate_embeddings(self, label):
        embedding = 0
        num_found_words = 0

        for word in label.split(' '):
            try:
                embedding += self.keyed_vectors.wv.word_vec(word)
                num_found_words += 1
            except KeyError:
                # log.warning('Coult not find word vector: %s', word)
                pass

        if num_found_words == 0:
            return np.zeros(300)
        else:
            return embedding / num_found_words

    def _calculate_label_embeddings(self, index_list):
        pd_label_embeddings = pd.DataFrame(
            index=index_list,
            columns=['preprocessed', 'vector'])

        pd_label_embeddings['preprocessed'] = self.fpm.preprocess_column(
            pd_label_embeddings.index.to_series(),
            load_model=True)

        pd_label_embeddings['vector'] = pd_label_embeddings['preprocessed'].apply(
            self._caculate_embeddings)

        return pd_label_embeddings

    def _calculate_siblings_score(self, pair):
        class_label, entity_label = pair[0], pair[1]
        siblings = self.candidate_classes_info[class_label][1]

        siblings_embedding = 0
        for sibling in siblings:
            siblings_embedding += self.pd_entity_label_embeddings.loc[sibling, 'vector']
        siblings_embedding /= len(siblings)

        entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

        score = self._calculate_cosine_similarity(siblings_embedding, entity_embeddings)

        return score

    def _calculate_parents_score(self, pair):
        class_label, entity_label = pair[0], pair[1]
        entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

        # print(class_label, entity_label)

        paths_list = self.candidate_classes_info[class_label][0]
        num_paths = len(paths_list)

        score = 0
        for i in range(num_paths):
            path = paths_list[i][::-1]
            path_depth = len(paths_list[i])

            numerator = 0
            denominator = 0
            for j in range(path_depth):
                parent_embeddings = self.pd_class_label_embeddings.loc[path[j], 'vector']
                similarity = self._calculate_cosine_similarity(parent_embeddings, entity_embeddings)

                numerator += similarity * math.exp((j+1)/(path_depth))
                denominator += math.exp((j+1)/(path_depth))

            score += (numerator / denominator)

        score /= num_paths

        return score

    def _calculate_initial_scores(self):
        if file_exists(self.initial_siblings_scores) and file_exists(self.initial_parents_scores):
            pd_siblings_scores = pd.read_csv(self.initial_siblings_scores, index_col=0)
            pd_parents_scores = pd.read_csv(self.initial_parents_scores, index_col=0)
        else:
            entity_class_pairs = list(itertools.product(
                self.candidate_classes_label,
                self.candidate_entities))

            # calculate siblings score
            log.info('Calculating siblings score...')

            t1 = time()
            with multiprocessing.Pool(processes=9, maxtasksperchild=1) as p:
                results = p.map(self._calculate_siblings_score, entity_class_pairs)
            t2 = time()

            log.info('Elapsed time for calculating siblings score: %.2f minutes', (t2-t1)/60)

            results = np.array(results).reshape(self.num_candidate_classes, self.num_candidate_entities)

            pd_siblings_scores = pd.DataFrame(
                results,
                index=self.candidate_classes_label,
                columns=self.candidate_entities)

            pd_siblings_scores.to_csv(self.initial_siblings_scores)

            # calculate parents score
            log.info('Calculating parents score...')

            t1 = time()
            with multiprocessing.Pool(processes=9, maxtasksperchild=1) as p:
                results = p.map(self._calculate_parents_score, entity_class_pairs)
            t2 = time()

            log.info('Elapsed time for calculating parents score: %.2f minutes', (t2-t1)/60)

            results = np.array(results).reshape(self.num_candidate_classes, self.num_candidate_entities)

            pd_parents_scores = pd.DataFrame(
                results,
                index=self.candidate_classes_label,
                columns=self.candidate_entities)

            pd_parents_scores.to_csv(self.initial_parents_scores)

        return pd_siblings_scores, pd_parents_scores

    def run_iteration(self):
        num_iterations = math.floor(self.num_candidate_entities / self.num_mapping_per_iteration)

        iteration = 0
        while len(self.candidate_entities) > 0:
            log.info('Updating scores. Iteration: %d/%d', iteration, num_iterations)

            pd_scores = self.alpha*self.pd_siblings_scores + (1-self.alpha)*self.pd_parents_scores

            num_scores = pd_scores.shape[0] * pd_scores.shape[1]
            pd_top_scores = pd_scores.stack().nlargest(num_scores).reset_index()
            pd_top_scores.columns = ['candidate class', 'candidate entity', 'score']
            pd_top_scores.drop_duplicates(subset='candidate entity', inplace=True)

            top_n_scores = list(zip(pd_top_scores['candidate class'], pd_top_scores['candidate entity']))
            top_n_scores = top_n_scores[0:self.num_mapping_per_iteration]

            classes_to_update = list(set([x[0] for x in top_n_scores]))
            entities_to_remove = list(set([x[1] for x in top_n_scores]))

            # populate skeleton using selected entity
            for (candidate_class, candidate_entity) in top_n_scores:
                self.candidate_classes_info[candidate_class][1].append(candidate_entity)

            if len(self.candidate_entities) <= self.num_mapping_per_iteration:
                break

            # remove selected entities from candidate entities and scores
            self.candidate_entities = list(set(self.candidate_entities) - set(entities_to_remove))
            self.pd_siblings_scores = self.pd_siblings_scores.drop(labels=entities_to_remove, axis=1)
            self.pd_parents_scores = self.pd_parents_scores.drop(labels=entities_to_remove, axis=1)

            # update siblings score
            entity_class_pairs = list(itertools.product(
                classes_to_update,
                self.candidate_entities))

            t1 = time()
            with multiprocessing.Pool(processes=9, maxtasksperchild=1) as p:
                results = p.map(self._calculate_siblings_score, entity_class_pairs)
            t2 = time()

            results = np.array(results).reshape(len(classes_to_update), len(self.candidate_entities))

            pd_siblings_to_update = pd.DataFrame(
                results,
                index=classes_to_update,
                columns=self.candidate_entities)

            self.pd_siblings_scores.update(pd_siblings_to_update)

            log.info('Elapsed time for updating scores: %.2f minutes', (t2-t1)/60)

            iteration += 1

        return self.candidate_classes_info
