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

    def __init__(self, stem_lookup_filepath):
        """
        Class initializer.

        Inputs:
        """
        self.pd_stem_lookup = pd.read_csv(
            stem_lookup_filepath,
            sep='\t',
            index_col='stemmed',
            keep_default_na=False)

    def decode_query(self, query):
        if '_' in query:
            elements = query.split('_')
            element_candidates = []

            if len(elements) > 2:
                raise ValueError('Unable to support n-grams where n > 2!')

            for element in elements:
                element_candidates.append(ast.literal_eval(
                    self.pd_stem_lookup.loc[element, 'originals']))

            candidates = {' '.join([k1, k2]): v1 * v2
                          for k1, v1 in element_candidates[0].items()
                          for k2, v2 in element_candidates[1].items()}
            candidates = {k: v for k, v in sorted(candidates.items(), key=lambda item: item[1], reverse=True)}
            candidates = list(candidates.keys())

        else:
            candidates = ast.literal_eval(self.pd_stem_lookup.loc[query, 'originals'])
            candidates = list(candidates.keys())

        return candidates

    def get_summary(self, queries, num_try=3, save_summaries=None, save_failed=None):
        summaries = []
        failed_queries = []

        num_queries = len(queries)
        log_every = [i * int(num_queries / NUM_LOGS) for i in range(NUM_LOGS)]

        for idx, query in enumerate(queries):
            if idx in log_every:
                log.info('Processing query {}/{}'.format(idx, num_queries))

            query_candidates = self.decode_query(query)

            success_flag = False
            for candidate in query_candidates[0:num_try]:
                try:
                    summary = wikipedia.WikipediaPage(candidate).summary.replace('\n', ' ')
                    summaries.append([query, candidate, summary])
                    success_flag = True
                    break
                except:
                    pass

            if not success_flag:
                failed_queries.append([query, ', '.join(query_candidates)])

        pd_summaries = pd.DataFrame(summaries, columns=['query', 'matching candidate', 'summary'])
        pd_failed = pd.DataFrame(failed_queries, columns=['query', 'failed candidates'])

        log.info('Successfully got wikipedia summaries for %d queries', pd_summaries.shape[0])
        log.info('Failed to get wikipedia summaries for %d queries', pd_failed.shape[0])

        if save_summaries:
            pd_summaries.to_csv(save_summaries, sep='\t', index=False)

        if save_failed:
            pd_failed.to_csv(save_failed, sep='\t', index=False)

        return pd_summaries, pd_failed


if __name__ == '__main__':
    wm = WikipediaManager()
    wm.get_summary(
        ['graham_cracker', 'chow_mein', 'dulg', 'chocol', 'dumpl'],
        save_summaries='/home/jyoun/Jason/Research/FoodOntology/output/summaries.txt',
        save_failed='/home/jyoun/Jason/Research/FoodOntology/output/failed.txt')
