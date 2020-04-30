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
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import pandas as pd
import numpy as np
import pickle


# local imports
from utils.config_parser import ConfigParser


def save_pkl(obj,save_to):
    """
    Pickle the object
    Inputs:
        obj: Object to pickle
        save_to : Filepath to pickle the object to
    """
    with open(save_to,'wb') as fid:
        pickle.dump(obj,fid)

def load_pkl(load_from):
    """
    Load the pickled object
    Inputs:
        load_from : Filepath to pickled object
    Output:
        obj: Pickled Object 
    """ 
    try:
        with open(load_from,'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except FileNotFoundError:
        return(0)

#Reference : https://www.python.org/doc/essays/graphs/
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

def get_parent_classes(foodonDF):
    parentList = list(foodonDF['Parent'])
    parentNP = np.array(parentList) 
    parentUnique = np.unique(parentNP)
    return(parentUnique)


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

def edit_label(labels_tmp):
    for idx,row in labels_tmp.iterrows():
        label=row['Preferred Label']
        if not('foodon' in label):
            label_replaced = label.replace('food product', '')
            label_replaced = label_replaced.replace(' food ', '')
            label_replaced = label_replaced.replace('food ', ' ')
            label_replaced = label_replaced.replace(' food ', ' ') 
            label_replaced = label_replaced.replace(' products ', ' ')
            label_replaced = label_replaced.replace('products ', ' ')
            label_replaced = label_replaced.replace(' products', ' ')   
            label_replaced = label_replaced.replace(' product ', ' ')
            label_replaced = label_replaced.replace('product ', ' ')
            label_replaced = label_replaced.replace(' product', ' ')
            
            row['Preferred Label'] = label_replaced 
    return(labels_tmp)


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
        self.fullontology_pkl = self.configparser.getstr('fullontology_pkl')
        self.skeleton_and_entities_pkl = self.configparser.getstr('skeleton_and_entities_pkl')
        self.overwrite_pkl = int(self.configparser.getstr('overwrite_pickle_flag'))
        self.foodonDF=pd.DataFrame()


    def get_parent_classes(self):
        parentList = list(self.foodonDF['Parent'])
        parentNP = np.array(parentList) 
        parentUnique = np.unique(parentNP)
        return(list(parentUnique))


    def get_classes(self,merge_min_count=2):
        """
        Get all candidate classes.
        """                

        # Check for previously saved pickle file
        ret_val = load_pkl(self.fullontology_pkl)
        if ret_val !=0: # Pickle file exists
            if self.overwrite_pkl !=1: # Do not create a new pickle file
                return(ret_val)
        
        # Read specified columns from FoodON.csv file         
        foodon=pd.read_csv(self.filepath,usecols =['Class ID','Parents','Preferred Label'])
        
        # Edit labels to remove occurences of 'food','product' and 'products'
        #Create dictionary of URI and ClassLabel
        labels_tmp = foodon[["Class ID", "Preferred Label"]].copy()
        labels_tmp=edit_label(labels_tmp)
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
        self.foodonDF = pd.DataFrame(pairs, columns=['Child', 'Parent'])

        # In foodonDF, replace URI by label
        for idx,pair in self.foodonDF.iterrows():
            pair['Child']=labels[pair['Child']]
            if pair['Parent'] in labels:
                pair['Parent']=labels[pair['Parent']]

        print('Parsed FoodON Ontology file.\n')

        
        # Check for previously saved pickle file
        ret_val = load_pkl(self.fullontology_pkl)
        if ret_val !=0: # Pickle file exists
            if self.overwrite_pkl !=1: # Do not create a new pickle file
                print('Return classes from previously saved pickle file.')
                return(ret_val)
                        

        print('Creating FoodON ground truth Ontology Structure\n')

        # Get candidate classes (Will be used to build the Skeleton version of FoodON)
        uniqueCandidates = get_candidate_classes(self.foodonDF)

        # Create a dictionary version -> child:all parents
        childparentdict = {k: g["Parent"].tolist() for k,g in self.foodonDF.groupby("Child")}

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
                childrenrows = self.foodonDF.loc[self.foodonDF['Parent'] == c]
                for index,row in childrenrows.iterrows():
                    child = row['Child']
                    children = children + [child]
                    uniquechildren = list(np.unique(np.array(children)))

                # Create new tuple with heirachypaths and children
                #  key = candidate , Value = tuple. 
                value_tuple = (paths_as_list_of_tuples,uniquechildren)
                candidate_dict[c] = value_tuple


        save_pkl(candidate_dict,self.fullontology_pkl)

        return candidate_dict

    def get_seeded_skeleton(self,candidate_dict,seed_count=2):
        ret_val = load_pkl(self.skeleton_and_entities_pkl)
        if ret_val !=0: 
            return(ret_val)

        entities_to_populate = []
        all_parents = self.get_parent_classes()

        for cd in candidate_dict.keys():
            value = candidate_dict[cd]
            paths = value[0]
            children = value[1]
            #entities = list(set(children) - set(key_list))
            entities = children
            if len(entities) > seed_count:
                seeds = random.choices(entities,k=seed_count)
            else:
                seeds=[random.choice(entities)]
            remaining_entities = list(set(entities) - set(seeds))
            remaining_entities = list(set(remaining_entities) - set(all_parents))
            update_value = (paths,seeds)
            candidate_dict[cd] = update_value
            entities_to_populate = entities_to_populate + remaining_entities
        
        return_tuple = (candidate_dict,entities_to_populate)
        save_pkl(return_tuple,self.skeleton_and_entities_pkl)
        return return_tuple


if __name__ == '__main__':
    parse_foodon = ParseFoodOn('../config/foodon_parse.ini')

    class_list = parse_foodon.get_classes()
    seeded_skeleton = parse_foodon.get_seeded_skeleton(class_list)
    print('End of code')
