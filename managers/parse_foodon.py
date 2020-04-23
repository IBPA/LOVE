"""
Authors:
    Jason Youn - jyoun@ucdavis.edu
    Tarini Naravane - tnaravane@ucdavis.edu 

Description:
    Parse FoodOn.

To-do:
"""
# standard imports
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import pandas as pd
import numpy as np


# local imports
from utils.config_parser import ConfigParser

#https://www.python.org/doc/essays/graphs/
def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not start in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

def get_candidate_classes(foodonDF):
    # Step 1 - All Parents
    parentList = list(foodonDF['Parent'])
    parentNP = np.array(parentList) 
    parentUnique = np.unique(parentNP)
    # Step 2 - ONLY children - ie entities
    childOnly = list(set(foodonDF['Child'])-set(parentUnique))
    # Step 3 - candidates (ie parents of the entities)
    candidates = []
    for c in childOnly:
        selectPair = foodonDF.loc[foodonDF['Child'] == c].to_numpy() 
        candidates.append(selectPair[0,1])
        uniqueCandidates = np.unique(np.array(candidates))
    return(uniqueCandidates)

def merge_up(foodonDF):
    parentList = list(foodonDF['Parent'])
    parentNP = np.array(parentList) 
    parentUnique = np.unique(parentNP)
    # Step 2 - ONLY children - ie entities
    childOnly = list(set(foodonDF['Child'])-set(parentUnique))
    # Step 3 - candidates (ie parents of the entities)
    candidates = []
    for c in childOnly:
        selectPair = foodonDF.loc[foodonDF['Child'] == c].to_numpy() 
        candidates.append(selectPair[0,1])
        uniqueCandidates = np.unique(np.array(candidates))

    min_count = 2
    consolidatedFoodon = []

    merged=1

    for cd in uniqueCandidates:
        selectPairs = foodonDF.loc[foodonDF['Parent'] == cd].to_numpy() #look up children
        cd_children = list(selectPairs[:,0])
        cd_parents = foodonDF.loc[foodonDF['Child'] == cd].to_numpy() #look up parents

        # merge-up can only be done if candidate's children does NOT have another candidate
        if len(selectPairs) <= min_count and (set(cd_children).issubset(set(childOnly))) and len(cd_parents)==1: 
            # delete pair candidate as a parent, 
            # insert a child-parent pair where - child is children of candidate and parent is the parent of candidate
            for row_parent in cd_parents: 
                parent = row_parent[1]
                for row_child in selectPairs: # for all the child-parent pairs for candidate
                    child = row_child[0]
                    consolidatedFoodon.append([child,parent])
                    merged = merged+1
        else: #copy all pairs of (child, candidate) as is
            for row in selectPairs:
                consolidatedFoodon.append([row[0],row[1]])

    consolidatedfoodonDF = pd.DataFrame(consolidatedFoodon, columns=['Child', 'Parent'])
    return consolidatedfoodonDF



class ParseFoodOn:
    """
    Class for parsing FoodOn.
    """

    def __init__(self, config_filepath):
        """
        Class initializer.

        Inputs:
            config_filepath: (str) Configuration filepath.
        """
        self.configparser = ConfigParser(config_filepath)

        # read configuration file
        self.filepath = self.configparser.getstr('filepath')

        print(self.filepath)

    

    def get_classes(self,merge_min_count=2):
        """
        Get all candidate classes.
        """
        # Read specified columns from FoodON.csv file         
        foodon=pd.read_csv(self.filepath,usecols =['Class ID','Parents','Preferred Label'])
        # Create dictionary of URI and ClassLabel
        labels_tmp = foodon[["Class ID", "Preferred Label"]].copy()
        labels=labels_tmp.set_index('Class ID')['Preferred Label'].to_dict()

        #Create data frame with columns - child and all its' parents
        foodonOrigDF = (foodon[["Class ID", "Parents"]].copy()).rename(columns={'Class ID': 'Child'})
        
        #Split above DF into pairs of Child-Parent 
        pairs = []
        for index,row in foodonOrigDF.iterrows():
            parents = str(row['Parents'])
            parentList = parents.split("|")
            for pClass in parentList:
                child = str(row['Child'])
                pairs.append([child,pClass])

        foodonDF = pd.DataFrame(pairs, columns=['Child', 'Parent'])
        # Replace all elements from URI to label
        for idx,pair in foodonDF.iterrows():
            pair['Child']=labels[pair['Child']]
            if pair['Parent'] in labels:
                pair['Parent']=labels[pair['Parent']]


        # Get candidate classes (Will be used to build the Skeleton version of FoodON)
        uniqueCandidates = get_candidate_classes(foodonDF)

        # Create a dictionary version -> child:all parents
        childparentdict = {k: g["Parent"].tolist() for k,g in foodonDF.groupby("Child")}

        #Merging-up 
        #foodonDF = merge_up(foodonDF)
        #uniqueCandidates = get_candidate_classes(foodonDF)

        
        # Creating the Skeleton verion of FoodON
        end = labels['http://purl.obolibrary.org/obo/FOODON_00001002']  
        candidate_dict = {}

        for c in uniqueCandidates:
            paths = find_all_paths(childparentdict,c,end)
            paths_as_list_of_tuples = []
            if paths != []:
                for p in paths:
                    paths_as_list_of_tuples.append(tuple(p)) 
                children = []
                childrenrows = foodonDF.loc[foodonDF['Parent'] == c]
                for index,row in childrenrows.iterrows():
                    child = row['Child']
                    children = children + [child]
                # Create new tuple with heirachypaths and children
                #  key = candidate , Value = tuple. 
                value_tuple = (paths_as_list_of_tuples,children)
                candidate_dict[c] = value_tuple

        candidateclass = labels['http://purl.obolibrary.org/obo/CHEBI_22470']
        print('Key is ',candidateclass)
        print('Value is ',candidate_dict[candidateclass])

        return candidate_dict

    



if __name__ == '__main__':
    parse_foodon = ParseFoodOn('../config/foodon_parse.ini')

    class_list = parse_foodon.get_classes()
