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

    

    def get_classes(self):
        """
        Get all candidate classes.
        """
        # Read FoodON csv file filtered for data on Child and Parent nodes only 
        # And create DF with pairs of child and parent
        foodonOrigDF=pd.read_csv("/Users/tarininaravane/Documents/FoodOntologyAI/FoodONparsed.txt",delimiter="\t")
        foodonDF = pd.DataFrame(columns = ["Child","Parent"])

        """
        for index,row in foodonOrigDF.iterrows():
            parents = str(row['Parents'])
            parentList = parents.split("|")
            
            for pClass in parentList:
                child = str(row['Child'])
                a_row = pd.Series([child,pClass])
                row_df = pd.DataFrame([a_row])
                foodonDF = pd.concat([row_df, foodonDF], ignore_index=True)
        """

        foodonDF=pd.read_csv("/Users/tarininaravane/Documents/FoodOntologyAI/FoodONchildparentpairs.txt",delimiter="\t")

        # Step 1 - Getting list of Parents
        parentList = list(foodonDF['Parent'])
        parentNP = np.array(parentList) 
        parentUnique = np.unique(parentNP)

        # Step 2 - ONLY children - ie entities
        childOnly = list(set(foodonDF['Child'])-set(parentUnique))

        # Step 3 - candidates (ie parents of the entities)
        # List of candidates, look up parent for childOnly from dataframe. 
        # DF converted to numpy , to remove the column names and index 
        candidates = []
        for c in childOnly:
            selectPair = foodonDF.loc[foodonDF['Child'] == c].to_numpy() 
            candidates.append(selectPair[0,1])
        uniqueCandidates = np.unique(np.array(candidates))

        # Step 4 make a dictionary from Orig FoodON DF. key- child , value -list of parents
        childparentdict={}

        for index,row in foodonOrigDF.iterrows():
            parents = str(row['Parents'])
            child = row['Child']
            parentList = parents.split("|")
            childparentdict[child] = parentList

        # Step 5 Create dictionary of 
        end = 'http://purl.obolibrary.org/obo/FOODON_00001002' 
        candidate_dict = {}

        for c in uniqueCandidates:
            paths = find_all_paths(childparentdict,c,end)
            if paths != []:
                children = []
                childrenrows = foodonDF.loc[foodonDF['Parent'] == c]
                for index,row in childrenrows.iterrows():
                    child = row['Child']
                    children = children + [child]
                # Create new tuple with heirachypaths and children
                # Add tuple as value for key = candidate
                value_tuple = (paths,children)
                candidate_dict[c] = value_tuple

        out = dict(list(candidate_dict.items())[0: 1])  
        
        # printing result   
        print("Dictionary is :\n " + str(out))  

        return candidate_dict

    



if __name__ == '__main__':
    parse_foodon = ParseFoodOn('../config/foodon_parse.ini')

    class_list = parse_foodon.get_classes()
