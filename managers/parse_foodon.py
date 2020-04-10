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

# local imports
from utils.config_parser import ConfigParser


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
        class_list = []

        return class_list


if __name__ == '__main__':
    parse_foodon = ParseFoodOn('../config/foodon_parse.ini')

    class_list = parse_foodon.get_classes()
