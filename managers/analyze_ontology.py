"""
Authors:
    Tarini Naravane - tnaravane@ucdavis.edu

Description:
    Load required files for analysis

To-do:
"""
# standard imports
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# local imports
from utils.config_parser import ConfigParser
from utils.utilities import load_pkl


class AnalyzeOntology:
    def __init__(self, config_filepath):
        configparser = ConfigParser(config_filepath)
        gt_ontology_filename = configparser.getstr('gt_entitymapping')
        self.gt_ontology = load_pkl(gt_ontology_filename)

    def get_stats(self, predictedDF, allow_distance=0):
        TP = 0
        FP = 0
        fp_list = []
        distance_distribution = []

        for idx, pair in predictedDF.iterrows():
            predicted_class = pair['Parent']
            gt_classes = []
            for key, value in self.gt_ontology.items():
                if pair['Child'] in value[1]:
                    gt_classes.append(key)

            distance_list = []
            # common_ancestor_list = []
            for gt_class in gt_classes:
                predicted_paths = [path[::-1] for path in self.gt_ontology[predicted_class][0]]
                gt_paths = [path[::-1] for path in self.gt_ontology[gt_class][0]]

                for pred_path in predicted_paths:
                    for gt_path in gt_paths:
                        common_path = set(pred_path).intersection(gt_path)
                        common_path = [c for c in gt_path if c in common_path]

                        distance = len(pred_path) + len(gt_path) - 2*len(common_path)
                        # common_ancestor = common_path[-1]

                        distance_list.append(distance)
                        # common_ancestor_list.append(common_ancestor)

            idx_shortest = distance_list.index(min(distance_list))
            shortest_distance = distance_list[idx_shortest]
            # shortest_common_ancestor = common_ancestor_list[idx_shortest]
            distance_distribution.append(shortest_distance)

            if shortest_distance <= allow_distance:
                TP += 1
            else:
                FP += 1
                fp_list.append((pair['Parent'], pair['Child']))

        return (TP, FP, fp_list, distance_distribution)

