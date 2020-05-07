"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Select the best combination from grid search results.

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import sys
import multiprocessing
import itertools
from time import time

# local imports
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging


def main():
    """
    Main function.
    """
    beta = [1.0, 1.25, 1.5, 1.75, 2.0]
    alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_mapping = [10, 25, 50, 75, 100, 150, 200]

    grid_search_combination = list(itertools.product(beta, alpha, num_mapping))
    files_list = ['populated_beta{}_alpha{}_N{}.pkl'.format(c[0], c[1], c[2])
                  for c in grid_search_combination]

    with multiprocessing.Pool(processes=9, maxtasksperchild=1) as p:
        results = p.map(_calculate_siblings_score, files_list)


if __name__ == '__main__':
    main()
