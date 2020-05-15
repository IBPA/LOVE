"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Lookup queries on wikipedia and save them.

To-do:
"""
# standard imports
import ast
import logging as log
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import pandas as pd
import wikipedia

NUM_LOGS = 50


class WikipediaManager:
    """
    """

    def __init__(self):
        """
        Class initializer.

        Inputs:
        """
        pass

    def get_summary(self, queries, prev_summary=None, prev_failed=None):

        if prev_summary and prev_failed:
            log.info('Reusing previous summaries.')

            pd_prev_summaries = pd.read_csv(prev_summary, sep='\t', keep_default_na=False)
            pd_prev_failed = pd.read_csv(prev_failed, sep='\t', keep_default_na=False)

            known_successful_queries = pd_prev_summaries['query'].tolist()
            known_failed_queries = pd_prev_failed['query'].tolist()
            known_queries = known_successful_queries + known_failed_queries

            depracated_queries = [q for q in known_queries if q not in queries]
            queries = [q for q in queries if q not in known_queries]

            log.info('Found %d deprecated queries', len(depracated_queries))
            log.info('Found %d new queries', len(queries))

        summaries = []
        failed_queries = []

        num_queries = len(queries)
        log_every = [i * int(num_queries / NUM_LOGS) for i in range(NUM_LOGS)]

        log.info(queries)

        for idx, query in enumerate(queries):
            if idx in log_every:
                log.info('Processing query {}/{}'.format(idx, num_queries))

            try:
                summary = wikipedia.page(query).content.replace('\n', ' ')
                summaries.append([query, summary])
            except:
                failed_queries.append([query])

        pd_summaries = pd.DataFrame(summaries, columns=['query', 'summary'])
        pd_failed = pd.DataFrame(failed_queries, columns=['query'])

        if prev_summary and prev_failed:
            pd_summaries = pd_summaries.append(pd_prev_summaries)
            pd_failed = pd_failed.append(pd_prev_failed)

            pd_summaries = pd_summaries[~pd_summaries['query'].isin(depracated_queries)]
            pd_failed = pd_failed[~pd_failed['query'].isin(depracated_queries)]

        log.info('Successfully got wikipedia summaries for %d queries', pd_summaries.shape[0])
        log.info('Failed to get wikipedia summaries for %d queries', pd_failed.shape[0])

        return pd_summaries, pd_failed


if __name__ == '__main__':
    wm = WikipediaManager()
    wm.get_summary(
        ['graham_cracker', 'chow_mein', 'dulg', 'chocol', 'dumpl'],
        save_summaries='/home/jyoun/Jason/Research/FoodOntology/output/summaries.txt',
        save_failed='/home/jyoun/Jason/Research/FoodOntology/output/failed.txt')
