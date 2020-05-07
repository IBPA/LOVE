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
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import pandas as pd
import numpy as np
import pickle


# local imports
from utils.config_parser import ConfigParser


def load_pkl(load_from):
    """
    Load the pickled object
    Inputs:
        load_from : Filepath to pickled object
    Output:
        obj: Pickled Object
    """
    try:
        with open(load_from, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except FileNotFoundError:
        return(0)
    
def get_entities(pairsDF):
    #Step 1
    #foodonDF=pd.read_csv("FoodONpairs.txt",delimiter="\t")
    parentList = list(pairsDF['Parent'])
    parentNP = np.array(parentList) 
    parentUnique = np.unique(parentNP)
    print('get_entities function',len(parentUnique))
    # Step 2 - ONLY children - ie leaf-nodes
    childOnly = list(set(pairsDF['Child'])-set(parentUnique))
    return(childOnly)
  


class AnalyseOntology:
    def __init__(self, config_filepath):
        configparser = ConfigParser(config_filepath)
        self.foodonpairs = configparser.getstr('foodonpairs')
        self.gt_entitymapping = configparser.getstr('gt_entitymapping')

        # Load data from files
        ground_truth = load_pkl(self.gt_entitymapping)
        self.gtDF = self._getPairsfromDict(ground_truth)
        self.foodonDF = pd.read_csv(self.foodonpairs, delimiter="\t")
        self.foodon_graph = {k: g["Parent"].tolist() for k, g in self.foodonDF.groupby("Child")}

    @staticmethod
    def get_entities(pairsDF):
        # Step 1
        # foodonDF=pd.read_csv("FoodONpairs.txt",delimiter="\t")
        parentList = list(pairsDF['Parent'])
        parentNP = np.array(parentList)
        parentUnique = np.unique(parentNP)
        print('get_entities function', len(parentUnique))
        # Step 2 - ONLY children - ie leaf-nodes
        childOnly = list(set(pairsDF['Child'])-set(parentUnique))

        return childOnly

    def _getPairsfromDict(self, candidate_dict):
        pairs = []

        for parent in candidate_dict.keys():
            value = candidate_dict[parent]
            entities = value[1]

            for e in entities:
                pairs.append([e, parent])

        pairsDF = pd.DataFrame(pairs, columns=['Child', 'Parent'])

            if ((self.gtDF['Child'] == child) & (self.gtDF['Parent'] == parent)).any():
                # simple check whether it exists in GT
                TP = TP+1
            else:
                FP = FP + 1
                fp_list.append([child, parent])
        return (TP, FP, fp_list)


if __name__ == '__main__':
    ao = AnalyseOntology('../config/analyze_ontology.ini')
    # self.populated_pairs = load_pkl('/Users/tarininaravane/Documents/FoodOntologyAI/pairs_by_iteration.pkl')
    mock_file = '/Users/tarininaravane/Documents/FoodOntologyAI/mock_pred_pairs.txt'
    pairsDF = pd.read_csv(mock_file, delimiter="\t")
    # Add loop here to go through all populated ontology files
    tp, fp, fp_list = ao.get_stats(pairsDF)
    print('TP is ', tp)
    print('FP is ', fp)
