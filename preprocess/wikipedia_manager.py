"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:

To-do:
"""
# standard imports
import logging as log
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import numpy as np
import pandas as pd
import wikipedia

# local imports
from utils.config_parser import ConfigParser

NUM_LOGS = 10

class WikipediaManager:
    """
    """

    def __init__(self, queries, delim='_'):
        """
        Class initializer.

        Inputs:
        """
        if delim:
            self.queries = [query.replace(delim, ' ') for query in queries]
        else:
            self.queries = queries

        self.queries = list(set(self.queries))
        print('Loaded {} quries'.format(len(self.queries)))

    def get_summary(self, save_summaries=None, save_failed=None):
        summaries = []
        failed_queries = []

        num_queries = len(self.queries)
        log_every = [(i + 1) * int(num_queries / NUM_LOGS) for i in range(NUM_LOGS)]

        for idx, query in enumerate(self.queries):
            if idx in log_every:
                print('Processing query {}/{}'.format(idx, num_queries))

            try:
                summary = wikipedia.WikipediaPage(query).summary.replace('\n', '')
                summaries.append([query, summary])
            except:
                suggestion = wikipedia.suggest(query)
                failed_queries.append([query, suggestion if suggestion else ''])

        pd_summaries = pd.DataFrame(summaries, columns=['query', 'summary'])
        pd_failed = pd.DataFrame(failed_queries, columns=['query', 'suggestion'])

        if save_summaries:
            pd_summaries.to_csv(save_summaries, sep='\t', index=False)

        if save_failed:
            pd_failed.to_csv(save_failed, sep='\t', index=False)

if __name__ == '__main__':
    wm = WikipediaManager(['chow_mein', 'dumpling', 'efefsfsefesd', 'swiss missy', 'obama'])
    wm.get_summary()
