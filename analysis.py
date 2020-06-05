"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Select the best combination from grid search results.

To-do:
"""
# standard imports
import itertools
import logging as log
import multiprocessing
from time import time
import sys
import random

# third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, ttest_ind, entropy

# local imports
from managers.analyze_ontology import AnalyzeOntology
from utils.set_logging import set_logging
from utils.utilities import load_pkl


classes_without_embedding = [
    'alfonsinos',
    'argyrosomus',
    'azabicycloalkane',
    'diplectrum',
    'odacidae',
    'polyazaalkane',
    'azaalkane',
    'homopolysaccharide',
    'parastromateus',
    'imidazopyrimidine',
    'cherax',
    'oligopeptide',
    'cobamides',
    'penshell',
    'zosterisessor',
    'carboxamide',
    'atractoscion',
    'pentitol',
    'dimethylxanthine',
    'corrinoid',
    'knipowitschia',
    'acanthistius',
    'seerfish',
    'yokan',
    'p-menthan-3-ol',
    'cilus',
    'archaeogastropod',
    'glycosylglucose',
    'mycteroperca',
    'cobblerfish',
    'varunidae',
    'larimichthys',
    'grayling',
    'cutlassfish',
    'galacturonan',
    'trimethylxanthine',
    'monodont',
    'cassidula',
    'tetritol',
    'mudskipper',
    'aminopyrimidine',
    'rhombosoleidae',
    'lateolabracidae',
    'pomatoschistus',
    'cirriped',
    'cephalopholis',
    'nerite',
    'pteridines',
    'dentex',
    'tetrapyrrole',
    'porcupinefish',
    'hexose',
    'branchiopod',
    'metallotetrapyrrole',
    'an',
    'mesogastropod',
    'aldrichetta',
    'harengula',
    'kingklip',
    'aldohexose',
    'jeotgal',
    'salmonins',
    'leatherjacket',
    'aldose',
    'genyonemus',
    'diazines',
    'alditol',
    'pterins',
    'moromi',
    'codling',
    'halibut',
    'neogastropod',
    'arkshells',
    'polypyrrole',
]

entities_without_embedding = [
    'sucralose',
    'azorubine, carmoisine',
    'mandoo',
    'moose',
    'ossobuco',
    'piki',
    'cowcod',
    'sandperch',
    'aonori',
    'nursehound',
    'thalipeeth bhajani',
    'hydroxocobalamin',
    'brownspotted rockcod',
    'acanthistius',
    'tsukudani',
    'matzoth',
    'sucroglycerides',
    'konyak',
    'catla',
    'dextrose',
    'eelpout',
    'thalipeeth',
    'kapenta',
    'lateolabracidae',
    'bon bon',
    '4-hexylresorcinol',
    'eulachon',
    'neohesperidine dc',
    'wreckfish',
    'hemicellulose',
    'Seongge-jeot',
    'schillerlocken',
    'ladyfish',
    'porae',
    'yakjoo',
    'hexamethylenetetramine',
    'arak',
    'beefalo',
    '2,5,7,8-tetramethyl-2-(4,8,12-trimethyltridecyl)-3,4-dihydro-2H-1-benzopyran-6-ol',
    'Biltong',
    'erythrosin',
    'tanmooji',
    'kenkey',
    'pirapitinga',
    'sneep',
    'bonnethead',
    'polkudu',
    'mulligatawny',
    'scallop made from surimi',
    'grayling',
    'galacturonan',
    'tarako',
    'shortjaw leatherjacket',
    'koshou',
    'humantin',
    'sandsmelt',
    'tanok',
    'nham',
    'polyvinylpolypyrrolidone',
    'petit four',
    'croziflette',
    'beiju',
    'tomacouli',
    'gehakt',
    'kyungyook',
    'tartiflette',
    'pigfish',
    'glucitol',
    'gayal',
    'weatherfish',
    'maltose',
    'baumkuchen',
    'squillid',
    'goosefish',
    'sucrose',
    'shipworm',
    'pastirma',
    'ratfish',
    'parastromateus',
    'cheesefurter',
    'tajine',
    'palometa',
    'taimen',
    'shortnose spurdog',
    'chapati',
    'moonfish',
    'capsanthin',
    'doujiang',
    'porkfish',
    'musli',
    'sandeel',
    'poult',
    'Glucuronolactone',
    'cassidula',
    'angola dentex',
    'skilfish',
    'tryptophan',
    'johnnycake',
    'anchoveta',
    'ikura',
    'pu dong',
    'dugong',
    'parevine',
    'polydextrose',
    'ox',
    'tarakihi',
    'bu-du',
    'poppadum',
    'megrim',
    'hogchoker',
    'nahrzucker',
    'Verhackert',
    'sanddab',
    'dexpantothenol',
    'tahkjoo',
    'duqqa',
    'twoenjang',
    'thiabendazole',
    'greenling',
    'searobin',
    'isada krill',
    'kinako',
    'sugee',
    'morwong',
    'mellorine',
    'pintadilla',
    'sujiko',
    'saridele',
    'cutlassfish',
    'akutaq',
    'harvestfish',
    'diphos sanguin',
    'sobrasada',
    'kahawai',
    'galactose',
    'nelma',
    'litholrubine bk',
    'advantame',
    'cachama',
    'biwa',
    'aldrichetta',
    'somiviki',
    'saithe',
    'minarine',
    'goldeye',
    'bakkoji',
    'surimi',
    'flyingfish',
    'tuaw jaew',
    'rohu',
    'buffalofish',
    'leatherjacket',
    'akee and saltfish',
    'choonjang',
    'yookpo',
    'kimchi',
]

wine_classes = [
    'red wine',
    'wine or wine-like food product',
    'black grape wine',
    'grape based wine or wine-like food product',
    'low-alcohol wine food product',
    'grape wine',
    'fruit wine',
    'grape based low-alcohol wine food product',
    'non-fruit wine food product',
    'white wine',
    'wine, 7-24% alcohol, food product',
    'heavy wine, 14-24% alcohol, food product',
    'light wine',
    'grape wine by region'
]

bean_classes = [
    'broad bean (whole)',
    'great northern bean substance',
    'great northern bean food product',
    'tempeh food product',
    'rice bean food product',
    'white bean (whole)',
    'broad bean food product',
    'green bean substance',
    'great northern bean (whole)',
    'yam bean food product',
    'navy bean food product',
    'bean food product',
    'bean (canned)',
    'castor bean food product',
    'mung bean food product',
    'bean pod',
    'cranberry bean (whole)',
    'pink bean (whole)',
    'jack bean (whole)',
    'kidney bean (whole)',
    'asparagus bean food product',
    'navy bean (whole)',
    'pink bean food product',
    'garbanzo bean substance',
    'lentil (whole)',
    'kidney bean substance',
    'garbanzo bean (whole)',
    'bean substance',
    'adzuki bean food product',
    'pinto bean food product',
    'fermented bean product',
    'wax bean food product',
    'dry bean food product',
    'kidney bean (whole, dried)',
    'chickpea (whole)',
    'velvet bean (whole)',
    'pea (whole, dried)',
    'green bean food product',
    'bean (cooked)',
    'bean (whole)',
    'miso food product',
    'black gram bean (whole)',
    'brown bean food product',
    'kidney bean food product',
    'moth bean food product',
    'winged bean food product',
    'hyacinth bean food product',
    'marrow bean (whole)',
    'pinto bean (whole)',
    'soybean (whole)',
    'tofu food product',
    'garbanzo bean (vegetable) food product',
    'infant formula (soy-based)',
    'fermented soybean food product',
    'common bean food product',
    'bean (whole, dried)',
    'bambara groundnut (whole)',
    'bean flour',
    'lima bean substance',
    'white bean food product',
    'dry pea food product',
    'red kidney bean (whole)',
    'scarlet runner bean food product',
    'adzuki bean substance',
    'mung bean (whole)',
    'edible bean pod',
    'chickpea food product',
    'bean sprout',
    'adzuki bean (whole)',
    'rice bean (whole)',
    'kidney bean (canned)',
    'cranberry bean food product',
    'soy based formula food product',
    'soybean paste',
    'bean (whole, raw)',
    'hyacinth bean (whole)',
    'lupine bean food product',
    'shell bean pod',
    'black turtle bean (whole)',
    'jack-bean food product',
    'lima bean food product',
    'lima bean (whole)',
    'soybean food product',
    'winged bean (whole)',
    'moth bean (whole)',
    'soybean substance'
]

cheese_classes = [
    'semihard cheese product',
    'pasteurized cheese food product',
    'samsoe cheese',
    'hard grating cheese food product',
    'ovine cheese food product',
    'bovine cheese food product',
    'cow milk processed cheese product',
    'cheddar cheese food product',
    'soft cheese food product',
    'parmesan cheese',
    'pasteurized cheese spread food product',
    'cottage cheese',
    'cheddar cheese',
    'hard cheese food product',
    'cow milk cured cheese food product',
    'pasteurized blended cheese food product',
    'semisoft cheese product',
    'uncured cheese food product',
    'granular cheese',
    'cured cheese food product',
    'cow milk hard cheese food product',
    'blue cheese food product',
    'pasteurized process cheese food product',
    'pasteurized process cheese spread food product',
    'grated cheese',
    'processed cheese food product',
    'cream cheese',
    'cheese (made from buffalo milk)',
    'colby cheese',
    'swiss cheese',
    'cheese fondue',
    'cow milk cheese',
    'cottage cheese (creamed)',
    'sheep milk cheese food product',
    'uncured cow milk cheese food product',
    'goat milk cheese food product',
    'emulsified cheese product',
    'cheese dip',
    'cheese food product',
    'asiago cheese',
    'cold-pack cheese food product',
    'gouda cheese',
    'caprine cheese food product'
]


def save_figure(fig, save_to):
    fig.savefig(save_to, bbox_inches='tight')


def calculate_precision(file):
    population_pairs = load_pkl(file)
    iterations = list(population_pairs.keys())
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    pairs = []
    for iteration in iterations:
        pairs.extend(population_pairs[iteration])
    pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
    pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
    pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

    tp, fp, _, _, _ = analyze_ontoloty.get_stats(pd_pairs)
    precision = tp / (tp + fp)

    print(file, ': ', precision)

    return (precision, file)


def calculate_distance(file):
    population_pairs = load_pkl(file)
    iterations = list(population_pairs.keys())
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    pairs = []
    for iteration in iterations:
        pairs.extend(population_pairs[iteration])
    pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
    pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
    pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

    _, _, _, _, distance_distribution = analyze_ontoloty.get_stats(pd_pairs)

    print(file, ': ', np.mean(distance_distribution))

    return (np.mean(distance_distribution), file)


def find_best_grid_search_result(alpha_list, num_mapping_list):
    grid_search_combination = list(itertools.product(alpha_list, num_mapping_list))
    files_list = ['./data/scores/pairs_alpha{}_N{}.pkl'.format(c[0], c[1])
                  for c in grid_search_combination]

    t1 = time()
    with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
        result = p.map(calculate_precision, files_list)
    t2 = time()

    log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    pd_result = pd.DataFrame(result, columns=['Precision', 'Filename'])
    pd_result.sort_values(by='Precision', ascending=False, inplace=True)
    pd_result.to_csv('./output/grid_search_result.txt', sep='\t', index=False)


def distance_all_models():
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    # # random
    # files_list = ['./data/scores/random/pairs_{}.pkl'.format(i) for i in range(1, 51)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distance, files_list)
    # t2 = time()

    # pd_random_average_distance = pd.DataFrame(result, columns=['Average Distance', 'Filename'])
    # pd_random_average_distance.to_csv('./output/random_average_distance.txt', sep='\t', index=False)

    # # jaccard
    # files_list = ['./data/scores/jaccard/pairs_{}.pkl'.format(i) for i in range(1, 51)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distance, files_list)
    # t2 = time()

    # pd_jaccard_average_distance = pd.DataFrame(result, columns=['Average Distance', 'Filename'])
    # pd_jaccard_average_distance.to_csv('./output/jaccard_average_distance.txt', sep='\t', index=False)

    # # hamming
    # files_list = ['./data/scores/hamming/pairs_{}.pkl'.format(i) for i in range(1, 51)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distance, files_list)
    # t2 = time()

    # pd_hamming_average_distance = pd.DataFrame(result, columns=['Average Distance', 'Filename'])
    # pd_hamming_average_distance.to_csv('./output/hamming_average_distance.txt', sep='\t', index=False)

    # # glove
    # files_list = ['./data/scores/glove/pairs_{}.pkl'.format(i) for i in range(1, 51)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distance, files_list)
    # t2 = time()

    # pd_glove_average_distance = pd.DataFrame(result, columns=['Average Distance', 'Filename'])
    # pd_glove_average_distance.to_csv('./output/glove_average_distance.txt', sep='\t', index=False)

    # # glove_wiki
    # files_list = ['./data/scores/glove_wiki/pairs_{}.pkl'.format(i) for i in range(1, 51)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distance, files_list)
    # t2 = time()

    # pd_glove_wiki_average_distance = pd.DataFrame(result, columns=['Average Distance', 'Filename'])
    # pd_glove_wiki_average_distance.to_csv('./output/glove_wiki_average_distance.txt', sep='\t', index=False)

    # # wiki
    # files_list = ['./data/scores/wiki/pairs_{}.pkl'.format(i) for i in range(1, 51)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distance, files_list)
    # t2 = time()

    # pd_wiki_average_distance = pd.DataFrame(result, columns=['Average Distance', 'Filename'])
    # pd_wiki_average_distance.to_csv('./output/wiki_average_distance.txt', sep='\t', index=False)

    # sys.exit()

    # plot
    pd_random_average_distance = pd.read_csv('./output/random_average_distance.txt', sep='\t')
    pd_jaccard_average_distance = pd.read_csv('./output/jaccard_average_distance.txt', sep='\t')
    pd_hamming_average_distance = pd.read_csv('./output/hamming_average_distance.txt', sep='\t')
    pd_glove_average_distance = pd.read_csv('./output/glove_average_distance.txt', sep='\t')
    pd_glove_wiki_average_distance = pd.read_csv('./output/glove_wiki_average_distance.txt', sep='\t')
    pd_wiki_average_distance = pd.read_csv('./output/wiki_average_distance.txt', sep='\t')

    print('random')
    print(np.mean(pd_random_average_distance['Average Distance'].tolist()))

    print('jaccard')
    print(np.mean(pd_jaccard_average_distance['Average Distance'].tolist()))

    print('hamming')
    print(np.mean(pd_hamming_average_distance['Average Distance'].tolist()))

    print('glove')
    print(np.mean(pd_glove_average_distance['Average Distance'].tolist()))

    print('glove_wiki')
    print(np.mean(pd_glove_wiki_average_distance['Average Distance'].tolist()))

    print('wiki')
    print(np.mean(pd_wiki_average_distance['Average Distance'].tolist()))


def plot_precision_all_models():
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    # # random
    # files_list = ['./data/scores/random/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_random_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random_precision.to_csv('./output/random_precision.txt', sep='\t', index=False)

    # # jaccard
    # files_list = ['./data/scores/jaccard/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_jaccard_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_jaccard_precision.to_csv('./output/jaccard_precision.txt', sep='\t', index=False)

    # # hamming
    # files_list = ['./data/scores/hamming/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_hamming_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_hamming_precision.to_csv('./output/hamming_precision.txt', sep='\t', index=False)

    # # glove
    # files_list = ['./data/scores/glove/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_glove_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_glove_precision.to_csv('./output/glove_precision.txt', sep='\t', index=False)

    # # glove_wiki
    # files_list = ['./data/scores/glove_wiki/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_glove_wiki_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_glove_wiki_precision.to_csv('./output/glove_wiki_precision.txt', sep='\t', index=False)

    # # wiki
    # files_list = ['./data/scores/wiki/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_wiki_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_wiki_precision.to_csv('./output/wiki_precision.txt', sep='\t', index=False)

    # # wiki euclidean
    # files_list = ['./data/scores/wiki_euclidean/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # pd_wiki_euclidean_precision = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_wiki_euclidean_precision.to_csv('./output/wiki_euclidean_precision.txt', sep='\t', index=False)

    # plot
    pd_random_precision = pd.read_csv('./output/random_precision.txt', sep='\t')
    pd_jaccard_precision = pd.read_csv('./output/jaccard_precision.txt', sep='\t')
    pd_hamming_precision = pd.read_csv('./output/hamming_precision.txt', sep='\t')
    pd_glove_precision = pd.read_csv('./output/glove_precision.txt', sep='\t')
    pd_glove_wiki_precision = pd.read_csv('./output/glove_wiki_precision.txt', sep='\t')
    pd_wiki_precision = pd.read_csv('./output/wiki_precision.txt', sep='\t')
    pd_wiki_euclidean_precision = pd.read_csv('./output/wiki_euclidean_precision.txt', sep='\t')

    print('random')
    print(np.mean(pd_random_precision['Precision'].tolist()))
    print(np.std(pd_random_precision['Precision'].tolist()))

    print('jaccard')
    print(np.mean(pd_jaccard_precision['Precision'].tolist()))
    print(np.std(pd_jaccard_precision['Precision'].tolist()))

    print('hamming')
    print(np.mean(pd_hamming_precision['Precision'].tolist()))
    print(np.std(pd_hamming_precision['Precision'].tolist()))

    print('glove')
    print(np.mean(pd_glove_precision['Precision'].tolist()))
    print(np.std(pd_glove_precision['Precision'].tolist()))

    print('glove_wiki')
    print(np.mean(pd_glove_wiki_precision['Precision'].tolist()))
    print(np.std(pd_glove_wiki_precision['Precision'].tolist()))

    print('wiki')
    print(np.mean(pd_wiki_precision['Precision'].tolist()))
    print(np.std(pd_wiki_precision['Precision'].tolist()))

    print('wiki_euclidean')
    print(np.mean(pd_wiki_euclidean_precision['Precision'].tolist()))
    print(np.std(pd_wiki_euclidean_precision['Precision'].tolist()))

    pd_precision = pd.concat([
        # pd_random_precision,
        pd_jaccard_precision,
        pd_hamming_precision,
        pd_glove_precision,
        pd_glove_wiki_precision,
        pd_wiki_precision,
        pd_wiki_euclidean_precision])

    def _extract_method(filename):
        if 'random/' in filename:
            return 'random'
        elif 'jaccard/' in filename:
            return 'Jaccard'
        elif 'hamming/' in filename:
            return 'Hamming'
        elif 'glove/' in filename:
            return 'GloVe'
        elif 'glove_wiki/' in filename:
            return 'GloVe_Wiki'
        elif 'wiki/' in filename:
            return 'Wiki'
        elif 'wiki_euclidean/' in filename:
            return 'Wiki_Euclidean'
        else:
            raise ValueError('Invalid filename: {}'.format(filename))

    pd_precision['Similarity Method'] = pd_precision['Filename'].apply(
        lambda x: _extract_method(x))

    fig = plt.figure()

    sns.set(style="whitegrid")

    ax = sns.boxplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['Jaccard', 'Hamming', 'GloVe', 'GloVe_Wiki', 'Wiki'])

    ax = sns.swarmplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['Jaccard', 'Hamming', 'GloVe', 'GloVe_Wiki', 'Wiki'])

    plt.axis([None, None, 0.08, 0.40])
    save_figure(fig, './output/different_models_precision.svg')

    # pairwise p-values
    _, pval = ttest_rel(pd_random_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('random vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_jaccard_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('jaccard vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_hamming_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('hamming vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_glove_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('glove vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_wiki_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('wiki vs. glove_wiki p-value: {}'.format(pval))


def plot_precision_with_without_food_product():
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    # # with
    # files_list = ['./data/scores/wiki/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_with = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_with.to_csv('./output/precision_with_food_product.txt', sep='\t', index=False)

    # # without
    # files_list = ['./data/scores/wiki_without_food_product/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_without = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_without.to_csv('./output/precision_without_food_product.txt', sep='\t', index=False)

    # plot
    pd_with = pd.read_csv('./output/precision_with_food_product.txt', sep='\t')
    pd_without = pd.read_csv('./output/precision_without_food_product.txt', sep='\t')

    print('with')
    print(np.mean(pd_with['Precision'].tolist()))
    print(np.std(pd_with['Precision'].tolist()))

    print('without')
    print(np.mean(pd_without['Precision'].tolist()))
    print(np.std(pd_without['Precision'].tolist()))

    pd_precision = pd.concat([
        pd_with,
        pd_without,
        ])

    def _extract_method(filename):
        if 'wiki/' in filename:
            return 'with'
        elif 'wiki_without' in filename:
            return 'without'
        else:
            raise ValueError('Invalid filename: {}'.format(filename))

    pd_precision['Similarity Method'] = pd_precision['Filename'].apply(
        lambda x: _extract_method(x))

    fig = plt.figure()

    sns.set(style="whitegrid")

    ax = sns.boxplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        # linewidth=1.0,
        # order=['random', 'jaccard', 'hamming', 'GloVe', 'GloVe_Wiki', 'Wiki'])
        order=['with', 'without'])

    ax = sns.swarmplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['with', 'without'])

    # plt.axis([None, None, 0.08, 0.40])
    plt.show()
    sys.exit()
    save_figure(fig, './output/different_models_precision.svg')

    # pairwise p-values
    _, pval = ttest_rel(pd_random_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('random vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_jaccard_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('jaccard vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_hamming_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('hamming vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_glove_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('glove vs. glove_wiki p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_wiki_precision['Precision'], pd_glove_wiki_precision['Precision'])
    print('wiki vs. glove_wiki p-value: {}'.format(pval))


def plot_precision_different_random_seeds():
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    # # num_seeds = 1
    # files_list = ['./data/scores/wiki/random_1/pairs_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random_1 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random_1.to_csv('./output/precision_num_seeds_1.txt', sep='\t', index=False)

    # # num_seeds = 2
    # files_list = ['./data/scores/wiki/random_2/pairs_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random_2 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random_2.to_csv('./output/precision_num_seeds_2.txt', sep='\t', index=False)

    # # num_seeds = 3
    # files_list = ['./data/scores/wiki/random_3/pairs_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random_3 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random_3.to_csv('./output/precision_num_seeds_3.txt', sep='\t', index=False)

    # # num_seeds = 4
    # files_list = ['./data/scores/wiki/random_4/pairs_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random_4 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random_4.to_csv('./output/precision_num_seeds_4.txt', sep='\t', index=False)

    # # num_seeds = 5
    # files_list = ['./data/scores/wiki/random_5/pairs_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random_5 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random_5.to_csv('./output/precision_num_seeds_5.txt', sep='\t', index=False)

    # sys.exit()

    # plot
    pd_random_1 = pd.read_csv('./output/precision_num_seeds_1.txt', sep='\t')
    pd_random_2 = pd.read_csv('./output/precision_num_seeds_2.txt', sep='\t')
    pd_random_3 = pd.read_csv('./output/precision_num_seeds_3.txt', sep='\t')
    pd_random_4 = pd.read_csv('./output/precision_num_seeds_4.txt', sep='\t')
    pd_random_5 = pd.read_csv('./output/precision_num_seeds_5.txt', sep='\t')

    print('seed = 1')
    print(np.mean(pd_random_1['Precision'].tolist()))
    print(np.std(pd_random_1['Precision'].tolist()))

    print('seed = 2')
    print(np.mean(pd_random_2['Precision'].tolist()))
    print(np.std(pd_random_2['Precision'].tolist()))

    print('seed = 3')
    print(np.mean(pd_random_3['Precision'].tolist()))
    print(np.std(pd_random_3['Precision'].tolist()))

    print('seed = 4')
    print(np.mean(pd_random_4['Precision'].tolist()))
    print(np.std(pd_random_4['Precision'].tolist()))

    print('seed = 5')
    print(np.mean(pd_random_5['Precision'].tolist()))
    print(np.std(pd_random_5['Precision'].tolist()))

    pd_precision = pd.concat([
        pd_random_1,
        pd_random_2,
        pd_random_3,
        pd_random_4,
        pd_random_5,
    ])

    def _extract_method(filename):
        if 'random_1/' in filename:
            return '1'
        elif 'random_2/' in filename:
            return '2'
        elif 'random_3/' in filename:
            return '3'
        elif 'random_4/' in filename:
            return '4'
        elif 'random_5/' in filename:
            return '5'
        else:
            raise ValueError('Invalid filename: {}'.format(filename))

    pd_precision['Similarity Method'] = pd_precision['Filename'].apply(
        lambda x: _extract_method(x))

    fig = plt.figure()

    sns.set(style="whitegrid")

    ax = sns.boxplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['1', '2', '3', '4', '5'])

    ax = sns.swarmplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['1', '2', '3', '4', '5'])

    # plt.axis([None, None, 0.08, 0.40])
    save_figure(fig, './output/random_seeds_precision.svg')

    # # pairwise p-values
    # _, pval = ttest_rel(pd_random_precision['Precision'], pd_glove_wiki_precision['Precision'])
    # print('random vs. glove_wiki p-value: {}'.format(pval))

    # _, pval = ttest_rel(pd_jaccard_precision['Precision'], pd_glove_wiki_precision['Precision'])
    # print('jaccard vs. glove_wiki p-value: {}'.format(pval))

    # _, pval = ttest_rel(pd_hamming_precision['Precision'], pd_glove_wiki_precision['Precision'])
    # print('hamming vs. glove_wiki p-value: {}'.format(pval))

    # _, pval = ttest_rel(pd_glove_precision['Precision'], pd_glove_wiki_precision['Precision'])
    # print('glove vs. glove_wiki p-value: {}'.format(pval))

    # _, pval = ttest_rel(pd_wiki_precision['Precision'], pd_glove_wiki_precision['Precision'])
    # print('wiki vs. glove_wiki p-value: {}'.format(pval))

def plot_alpha_vs_precision():
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    # # alpha = 0.0
    # files_list = ['./data/scores/wiki/random_2/alpha_0.0/pairs_alpha0.0_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha0 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha0.to_csv('./output/precision_alpha_0.0.txt', sep='\t', index=False)

    # # alpha = 0.1
    # files_list = ['./data/scores/wiki/random_2/alpha_0.1/pairs_alpha0.1_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha1 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha1.to_csv('./output/precision_alpha_0.1.txt', sep='\t', index=False)

    # # alpha = 0.2
    # files_list = ['./data/scores/wiki/random_2/alpha_0.2/pairs_alpha0.2_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha2 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha2.to_csv('./output/precision_alpha_0.2.txt', sep='\t', index=False)

    # # alpha = 0.3
    # files_list = ['./data/scores/wiki/random_2/alpha_0.3/pairs_alpha0.3_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha3 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha3.to_csv('./output/precision_alpha_0.3.txt', sep='\t', index=False)

    # # alpha = 0.4
    # files_list = ['./data/scores/wiki/random_2/alpha_0.4/pairs_alpha0.4_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha4 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha4.to_csv('./output/precision_alpha_0.4.txt', sep='\t', index=False)

    # # alpha = 0.5
    # files_list = ['./data/scores/wiki/random_2/alpha_0.5/pairs_alpha0.5_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha5 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha5.to_csv('./output/precision_alpha_0.5.txt', sep='\t', index=False)

    # # alpha = 0.6
    # files_list = ['./data/scores/wiki/random_2/alpha_0.6/pairs_alpha0.6_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha6 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha6.to_csv('./output/precision_alpha_0.6.txt', sep='\t', index=False)

    # # alpha = 0.7
    # files_list = ['./data/scores/wiki/random_2/alpha_0.7/pairs_alpha0.7_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha7 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha7.to_csv('./output/precision_alpha_0.7.txt', sep='\t', index=False)

    # # alpha = 0.8
    # files_list = ['./data/scores/wiki/random_2/alpha_0.8/pairs_alpha0.8_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha8 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha8.to_csv('./output/precision_alpha_0.8.txt', sep='\t', index=False)

    # # alpha = 0.9
    # files_list = ['./data/scores/wiki/random_2/alpha_0.9/pairs_alpha0.9_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha9 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha9.to_csv('./output/precision_alpha_0.9.txt', sep='\t', index=False)

    # # alpha = 1.0
    # files_list = ['./data/scores/wiki/random_2/alpha_1.0/pairs_alpha1.0_{}.pkl'.format(i) for i in range(1, 101)]

    # t1 = time()
    # with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
    #     result = p.map(calculate_precision, files_list)
    # t2 = time()

    # log.info('Finished analysis. Elapsed time: %.02f min', (t2-t1)/60)

    # pd_random2_alpha10 = pd.DataFrame(result, columns=['Precision', 'Filename'])
    # pd_random2_alpha10.to_csv('./output/precision_alpha_1.0.txt', sep='\t', index=False)

    # sys.exit()

    # plot
    pd_random2_alpha0 = pd.read_csv('./output/precision_alpha_0.0.txt', sep='\t')
    pd_random2_alpha1 = pd.read_csv('./output/precision_alpha_0.1.txt', sep='\t')
    pd_random2_alpha2 = pd.read_csv('./output/precision_alpha_0.2.txt', sep='\t')
    pd_random2_alpha3 = pd.read_csv('./output/precision_alpha_0.3.txt', sep='\t')
    pd_random2_alpha4 = pd.read_csv('./output/precision_alpha_0.4.txt', sep='\t')
    pd_random2_alpha5 = pd.read_csv('./output/precision_alpha_0.5.txt', sep='\t')
    pd_random2_alpha6 = pd.read_csv('./output/precision_alpha_0.6.txt', sep='\t')
    pd_random2_alpha7 = pd.read_csv('./output/precision_alpha_0.7.txt', sep='\t')
    pd_random2_alpha8 = pd.read_csv('./output/precision_alpha_0.8.txt', sep='\t')
    pd_random2_alpha9 = pd.read_csv('./output/precision_alpha_0.9.txt', sep='\t')
    pd_random2_alpha10 = pd.read_csv('./output/precision_alpha_1.0.txt', sep='\t')

    print('alpha = 0.0')
    print(np.mean(pd_random2_alpha0['Precision'].tolist()))
    print(np.std(pd_random2_alpha0['Precision'].tolist()))

    print('alpha = 0.1')
    print(np.mean(pd_random2_alpha1['Precision'].tolist()))
    print(np.std(pd_random2_alpha1['Precision'].tolist()))

    print('alpha = 0.2')
    print(np.mean(pd_random2_alpha2['Precision'].tolist()))
    print(np.std(pd_random2_alpha2['Precision'].tolist()))

    print('alpha = 0.3')
    print(np.mean(pd_random2_alpha3['Precision'].tolist()))
    print(np.std(pd_random2_alpha3['Precision'].tolist()))

    print('alpha = 0.4')
    print(np.mean(pd_random2_alpha4['Precision'].tolist()))
    print(np.std(pd_random2_alpha4['Precision'].tolist()))

    print('alpha = 0.5')
    print(np.mean(pd_random2_alpha5['Precision'].tolist()))
    print(np.std(pd_random2_alpha5['Precision'].tolist()))

    print('alpha = 0.6')
    print(np.mean(pd_random2_alpha6['Precision'].tolist()))
    print(np.std(pd_random2_alpha6['Precision'].tolist()))

    print('alpha = 0.7')
    print(np.mean(pd_random2_alpha7['Precision'].tolist()))
    print(np.std(pd_random2_alpha7['Precision'].tolist()))

    print('alpha = 0.8')
    print(np.mean(pd_random2_alpha8['Precision'].tolist()))
    print(np.std(pd_random2_alpha8['Precision'].tolist()))

    print('alpha = 0.9')
    print(np.mean(pd_random2_alpha9['Precision'].tolist()))
    print(np.std(pd_random2_alpha9['Precision'].tolist()))

    print('alpha = 1.0')
    print(np.mean(pd_random2_alpha10['Precision'].tolist()))
    print(np.std(pd_random2_alpha10['Precision'].tolist()))

    pd_precision = pd.concat([
        pd_random2_alpha0,
        pd_random2_alpha1,
        pd_random2_alpha2,
        pd_random2_alpha3,
        pd_random2_alpha4,
        pd_random2_alpha5,
        pd_random2_alpha6,
        pd_random2_alpha7,
        pd_random2_alpha8,
        pd_random2_alpha9,
        pd_random2_alpha10,
    ])

    def _extract_method(filename):
        if 'alpha_0.0/' in filename:
            return '0.0'
        elif 'alpha_0.1/' in filename:
            return '0.1'
        elif 'alpha_0.2/' in filename:
            return '0.2'
        elif 'alpha_0.3/' in filename:
            return '0.3'
        elif 'alpha_0.4/' in filename:
            return '0.4'
        elif 'alpha_0.5/' in filename:
            return '0.5'
        elif 'alpha_0.6/' in filename:
            return '0.6'
        elif 'alpha_0.7/' in filename:
            return '0.7'
        elif 'alpha_0.8/' in filename:
            return '0.8'
        elif 'alpha_0.9/' in filename:
            return '0.9'
        elif 'alpha_1.0/' in filename:
            return '1.0'
        else:
            raise ValueError('Invalid filename: {}'.format(filename))

    pd_precision['Similarity Method'] = pd_precision['Filename'].apply(
        lambda x: _extract_method(x))

    fig = plt.figure()

    sns.set(style="whitegrid")

    ax = sns.boxplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    ax = sns.swarmplot(
        x='Similarity Method',
        y='Precision',
        data=pd_precision,
        order=['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    # plt.axis([None, None, 0.08, 0.40])
    save_figure(fig, './output/alpha_vs_precision.svg')

    # pairwise p-values
    _, pval = ttest_rel(pd_random2_alpha6['Precision'], pd_random2_alpha7['Precision'])
    print('pd_random2_alpha6 vs. pd_random2_alpha7 p-value: {}'.format(pval))

    _, pval = ttest_rel(pd_random2_alpha7['Precision'], pd_random2_alpha8['Precision'])
    print('pd_random2_alpha7 vs. pd_random2_alpha8 p-value: {}'.format(pval))


def calculate_distribution(filename):
    print(filename)
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    population_pairs = load_pkl(filename)
    iterations = list(population_pairs.keys())
    iterations.sort()

    pairs = []
    for iteration in iterations:
        pairs.extend(population_pairs[iteration])
    pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
    pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
    pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

    _, _, _, _, distance_distribution = analyze_ontoloty.get_stats(pd_pairs)

    distance_dict = {}
    for distance in distance_distribution:
        if distance not in distance_dict:
            distance_dict[distance] = 1
        else:
            distance_dict[distance] += 1

    distance_list = list(distance_dict.keys())
    samples_list = list(distance_dict.values())

    pd_concat = pd.DataFrame(
        {'Distance': distance_list,
         '# of TPs': samples_list})

    return pd_concat

def plot_distance_vs_precision_and_TPs():
    # # random
    # files_list = ['./data/scores/random/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=8, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distribution, files_list)
    # t2 = time()
    # print(t2-t1)

    # pd_random = pd.concat(result)
    # pd_random.to_csv('./output/random_distance_num_TPs.txt', sep='\t', index=False)
    pd_random = pd.read_csv('./output/random_distance_num_TPs.txt', sep='\t')
    print(pd_random.shape)

    # # glove wiki
    # files_list = ['./data/scores/glove_wiki/pairs_{}.pkl'.format(i) for i in range(1, 101)]
    # t1 = time()
    # with multiprocessing.Pool(processes=8, maxtasksperchild=1) as p:
    #     result = p.map(calculate_distribution, files_list)
    # t2 = time()
    # print(t2-t1)

    # pd_glove_wiki = pd.concat(result)
    # pd_glove_wiki.to_csv('./output/glove_wiki_distance_num_TPs.txt', sep='\t', index=False)
    pd_glove_wiki = pd.read_csv('./output/glove_wiki_distance_num_TPs.txt', sep='\t')
    print(pd_glove_wiki.shape)

    # statistics
    random_distribution = []
    for idx, row in pd_random.iterrows():
        distance = row['Distance']
        num_TPs = row['# of TPs']
        random_distribution.extend([distance for _ in range(num_TPs)])

    glove_wiki_distribution = []
    for idx, row in pd_glove_wiki.iterrows():
        distance = row['Distance']
        num_TPs = row['# of TPs']
        glove_wiki_distribution.extend([distance for _ in range(num_TPs)])

    entropy = ttest_ind(random_distribution, glove_wiki_distribution)
    print(entropy)
    sys.exit()

    # precision
    population_pairs = load_pkl('./data/scores/glove_wiki/pairs.pkl')
    iterations = list(population_pairs.keys())
    iterations.sort()

    # glove_wiki distance vs TPs
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')
    distance_distribution = pd_random['Distance'].tolist()
    precision_list = []
    precision_list = [0.336745406824147, 0.4692913385826772, 0.573490813648294, 0.6460629921259843, 0.7090551181102362, 0.763254593175853, 0.8146981627296588, 0.8644356955380578, 0.9150918635170604, 0.9524934383202099, 0.9750656167979003, 0.9893700787401575, 0.99501312335958, 0.9993438320209974, 0.9998687664041995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # for allow_distance in range(max(distance_distribution)+1):
    #     log.info('Processing allowed distance: %d/%d',
    #              allow_distance, max(distance_distribution))

    #     pairs = []
    #     for iteration in iterations:
    #         pairs.extend(population_pairs[iteration])
    #     pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
    #     pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
    #     pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

    #     if allow_distance > 16:
    #         precision_list.append(1.0)
    #     else:
    #         tp, fp, _, _, _ = analyze_ontoloty.get_stats(pd_pairs, allow_distance=allow_distance)
    #         precision_list.append(tp / (tp + fp))

    #     print(precision_list)

    pd_precision = pd.DataFrame(
        {'Distance': list(range(max(distance_distribution)+1)),
         'Precision': precision_list})

    # random distance vs TPs
    population_pairs = load_pkl('./data/scores/random/pairs_1.pkl')
    iterations = list(population_pairs.keys())
    iterations.sort()

    distance_distribution = pd_random['Distance'].tolist()
    precision_list = []
    precision_list = [0.0006669334400426837, 0.003201280512204882, 0.01027077497665733, 0.024409763905562223, 0.05815659597172202, 0.12324929971988796, 0.23169267707082833, 0.38268640789649194, 0.5503534747232226, 0.7032146191810057, 0.8304655195411498, 0.916499933306656, 0.9610510871015072, 0.9830598906229159, 0.9929305055355475, 0.9967987194877951, 0.9985327464319061, 0.9991996798719488, 0.9991996798719488, 0.9997332266239829, 0.9998666133119914, 0.9998666133119914, 0.9998666133119914, 1.0, 1.0, 1.0]
    for allow_distance in range(max(distance_distribution)+1):
        log.info('Processing allowed distance: %d/%d',
                 allow_distance, max(distance_distribution))

        pairs = []
        for iteration in iterations:
            pairs.extend(population_pairs[iteration])
        pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
        pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
        pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

        tp, fp, _, _, _ = analyze_ontoloty.get_stats(pd_pairs, allow_distance=allow_distance)
        precision_list.append(tp / (tp + fp))

        print(precision_list)

    pd_random_precision = pd.DataFrame(
        {'Distance': list(range(max(distance_distribution)+1)),
         'Precision': precision_list})

    # figure
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    sns.lineplot(
        x='Distance',
        y='Precision',
        data=pd_precision,
        color='black',
        ax=ax1)

    sns.lineplot(
        x='Distance',
        y='Precision',
        data=pd_random_precision,
        color='gray',
        ax=ax1)

    sns.barplot(
        x='Distance',
        y='# of TPs',
        data=pd_glove_wiki,
        color='black',
        saturation=.5,
        alpha=0.8,
        ci=None,
        ax=ax2,
        zorder=2)

    sns.barplot(
        x='Distance',
        y='# of TPs',
        data=pd_random,
        color='gray',
        saturation=.5,
        alpha=0.8,
        ci=None,
        ax=ax2)

    plt.grid(True)
    # plt.show()
    save_figure(fig, './output/distance_vs_precision_and_tps.svg')


def plot_foodon_analysis():
    # # pie chart fig 4
    # fig, ax = plt.subplots()

    # size = 0.3
    # vals = np.array([[331., 2433.], [3111., 7754.]])

    # cmap = plt.get_cmap("tab20c")
    # outer_colors = cmap(np.arange(3)*4)
    # inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

    # labels_outer = [
    #     'Classes\n(2,764)',
    #     'Entities\n(10,865)']

    # ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
    #        wedgeprops=dict(width=size, edgecolor='w'), labels=labels_outer)

    # labels_inner = [
    #     'Non-candidate class\n(331)',
    #     'Candidate class\n(2,433)',
    #     'Seed entity\n(3,111)',
    #     'Candidate entity\n(7,754)']

    # ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
    #        wedgeprops=dict(width=size, edgecolor='w'), labels=labels_inner, labeldistance=0.5)

    # ax.set(aspect='equal')
    # save_figure(fig, './output/foodon_class_entity_pie_chart.svg')

    # pie chart data
    ontology_dict = load_pkl('./data/FoodOn/full_classes_dict.pkl')
    class_labels = list(ontology_dict.keys())

    level_1_dict = {}
    for class_label in class_labels:
        for path in ontology_dict[class_label][0]:
            if len(path) == 1:
                break

            if path[-2] not in level_1_dict:
                level_1_dict[path[-2]] = [class_label]
            else:
                level_1_dict[path[-2]].append(class_label)

    keys = list(level_1_dict.keys())
    keys.sort()

    level_1_dict_sorted = {}
    for key in keys:
        level_1_dict_sorted[key] = len(set(level_1_dict[key]))

    # fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))
    # wedges, texts = ax.pie(level_1_dict_sorted.values(), wedgeprops=dict(width=0.5), startangle=-40)
    # kw = dict(arrowprops=dict(arrowstyle="-"), bbox=None, zorder=0, va="center")

    # for i, p in enumerate(wedges):
    #     ang = (p.theta2 - p.theta1)/2. + p.theta1
    #     y = np.sin(np.deg2rad(ang))
    #     x = np.cos(np.deg2rad(ang))
    #     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    #     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    #     kw["arrowprops"].update({"connectionstyle": connectionstyle})
    #     ax.annotate(list(level_1_dict_sorted.keys())[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
    #                 horizontalalignment=horizontalalignment, **kw)

    fig, ax = plt.subplots()

    size = 0.3
    vals = np.array([1838, 270, 252, 225, 94, 59, 117])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)

    # 'food product by organism' = 1838
    # 'food component' = 270
    # 'food product by process' = 252
    # 'food product by meal type' = 225
    # 'food product by quality' = 94
    # 'prepared food product' = 59
    # 'multi-component food' = 50
    # 'food product analog' = 38
    # 'food product by consumer group' = 18
    # 'processed food product' = 9
    # 'biotechnologically derived food' = 1
    # 'cultural food product' = 1

    labels = [
        'food product by organism',
        'food component',
        'food product by process',
        'food product by meal type',
        'food product by quality',
        'prepared food product',
        'others']

    wedges, _ = ax.pie(
        vals,
        radius=1,
        wedgeprops=dict(width=0.5),
        colors=outer_colors,
        startangle=218)

    kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    save_figure(fig, './output/foodon_level_1_pie_chart.svg')
    sys.exit()

    # number of paths to root
    ontology_dict = load_pkl('./data/FoodOn/candidate_classes_dict.pkl')
    class_labels = list(ontology_dict.keys())

    number_of_paths_dict = {}
    for class_label in class_labels:
        paths = ontology_dict[class_label][0]
        num_paths = len(paths)

        if num_paths not in number_of_paths_dict:
            number_of_paths_dict[num_paths] = 1
        else:
            number_of_paths_dict[num_paths] += 1

    keys = list(number_of_paths_dict.keys())
    keys.sort()

    number_of_paths_dict_sorted = {}
    for key in keys:
        number_of_paths_dict_sorted[key] = number_of_paths_dict[key]

    fig = plt.figure()
    plt.bar(number_of_paths_dict_sorted.keys(), number_of_paths_dict_sorted.values())
    plt.xlabel('Number of paths to the root')
    plt.ylabel('Number of candidate classes')
    plt.yscale('log')
    plt.xticks(list(number_of_paths_dict_sorted.keys()))
    save_figure(fig, './output/foodon_number_of_paths_vs_classes.svg')

    # number of entities in each class
    ontology_dict = load_pkl('./data/FoodOn/full_classes_dict.pkl')
    class_labels = list(ontology_dict.keys())

    entities_number_dict = {}
    for class_label in class_labels:
        entities = ontology_dict[class_label][1]
        num_entities = len(set(entities))

        if num_entities not in entities_number_dict:
            entities_number_dict[num_entities] = 1
        else:
            entities_number_dict[num_entities] += 1

    keys = list(entities_number_dict.keys())
    keys.sort()

    entities_number_dict_sorted = {}
    for key in keys:
        entities_number_dict_sorted[key] = entities_number_dict[key]

    fig = plt.figure(figsize=(30, 10))
    plt.bar(entities_number_dict_sorted.keys(), entities_number_dict_sorted.values())
    plt.xlabel('Number of entities in each class')
    plt.ylabel('Number of classes')
    plt.yscale('log')
    save_figure(fig, './output/foodon_num_entities_vs_num_classes.svg')

    # number of classses in each level
    ontology_dict = load_pkl('./data/FoodOn/full_classes_dict.pkl')
    class_labels = list(ontology_dict.keys())

    level_label_dict = {}
    for class_label in class_labels:
        for path in ontology_dict[class_label][0]:
            level = (len(path) - 1)
            if level not in level_label_dict:
                level_label_dict[level] = [class_label]
            else:
                level_label_dict[level].append(class_label)

    all_levels = list(level_label_dict.keys())
    all_levels.sort()

    level_number_dict = {}
    for level in all_levels:
        level_number_dict[level] = len(set(level_label_dict[level]))

    fig = plt.figure()
    plt.bar(level_number_dict.keys(), level_number_dict.values())
    plt.xlabel('Ontology depth')
    plt.ylabel('Number of classes')
    plt.yscale('log')
    plt.xticks(list(level_number_dict.keys()))
    save_figure(fig, './output/foodon_level_vs_num_classes.svg')


def visualize_foodon():
    pd_data = pd.read_csv('./data/FoodOn/foodonpairs.txt', sep='\t')

    all_classes = pd_data['Parent'].tolist()
    all_classes = list(set(all_classes))
    entities_dict = {k: v for v, k in enumerate(all_classes)}

    G = nx.DiGraph()

    for _, row in pd_data.iterrows():
        if row['Parent'] in all_classes and row['Child'] in all_classes:
            node_from = entities_dict[row['Parent']]
            node_to = entities_dict[row['Child']]
            G.add_edge(node_from, node_to)

    fig = plt.figure(figsize=(30, 8))
    pos = graphviz_layout(G, prog='dot')
    nx.draw(
        G,
        pos,
        with_labels=False,
        arrows=True,
        arrowsize=5,
        node_size=15,
        width=0.3)
    save_figure(fig, './output/foodon_visualization.svg')


def visualize_foodon_bean():
    pd_data = pd.read_csv('./data/FoodOn/foodonpairs.txt', sep='\t')

    all_classes = pd_data['Parent'].tolist()
    all_classes = list(set(all_classes))
    entities_dict = {k: v for v, k in enumerate(all_classes)}
    reverse_dict = {v: k for v, k in enumerate(all_classes)}

    G = nx.DiGraph()

    for _, row in pd_data.iterrows():
        if row['Parent'] in all_classes and row['Child'] in all_classes:
            node_from = entities_dict[row['Parent']]
            node_to = entities_dict[row['Child']]
            G.add_edge(node_from, node_to)

    bean_classes = []
    full_classes_dict = load_pkl('./data/FoodOn/full_classes_dict.pkl')
    for key, val in full_classes_dict.items():
        for path in val[0]:
            if 'bean food product' in path:
                bean_classes.append(key)
    bean_classes = list(set(bean_classes))
    print(bean_classes)
    sys.exit()

    wine_classes = []
    full_classes_dict = load_pkl('./data/FoodOn/full_classes_dict.pkl')
    for key, val in full_classes_dict.items():
        for path in val[0]:
            if 'wine or wine-like food product' in path:
                wine_classes.append(key)
    wine_classes = list(set(wine_classes))
    print(wine_classes)
    sys.exit()

    G_bean = G.subgraph([entities_dict[bean_class] for bean_class in bean_classes])


    labels = {}
    for node in G_bean.nodes():
        if node in [entities_dict[bean_class] for bean_class in bean_classes]:
            # labels[node] = reverse_dict[node]
            labels[node] = node

    fig = plt.figure(figsize=(30, 3))
    pos = graphviz_layout(G_bean, prog='dot')

    nx.draw(G_bean, pos, with_labels=False, node_color='r')
    nx.draw_networkx_labels(G_bean, pos, labels, font_size=5)

    plt.show()
    # fig = plt.figure(figsize=(30, 8))
    # pos = graphviz_layout(G, prog='dot')
    # nx.draw(
    #     G_bean,
    #     pos,
    #     with_labels=True,
    #     arrows=True)
    #     # arrowsize=5,
    #     # node_size=15,
    #     # width=0.3)

    save_figure(fig, './output/bean_ontology_visualization.svg')


def cheese_bean_wine_precision():
    population_pairs = load_pkl('./data/scores/wiki/pairs_21.pkl')
    iterations = list(population_pairs.keys())
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    pairs = []
    for iteration in iterations:
        pairs.extend(population_pairs[iteration])
    pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
    pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
    pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

    columns = ['Ground Truth Classes', 'Predicted Class', 'Candidate Entity']

    # cheese
    tp, fp, tp_list, fp_list, _ = analyze_ontoloty.get_stats(pd_pairs, match_only=cheese_classes)
    precision = tp / (tp + fp)

    print('cheese precision: {}'.format(precision))
    print('cheese tp/fp: {}/{}'.format(tp, fp))

    pd_tps = pd.DataFrame(tp_list, columns=columns)
    pd_tps['Result'] = 'TP'
    pd_fps = pd.DataFrame(fp_list, columns=columns)
    pd_fps['Result'] = 'FP'
    pd_prediction = pd.concat([pd_tps, pd_fps])
    pd_prediction.to_csv('./output/cheese_prediction_result.csv', sep='\t', index=False)

    # # bean
    # tp, fp, tp_list, fp_list, _ = analyze_ontoloty.get_stats(pd_pairs, match_only=bean_classes)
    # precision = tp / (tp + fp)

    # print('bean precision: {}'.format(precision))
    # print('bean tp/fp: {}/{}'.format(tp, fp))

    # pd_tps = pd.DataFrame(tp_list, columns=columns)
    # pd_tps['Result'] = 'TP'
    # pd_fps = pd.DataFrame(fp_list, columns=columns)
    # pd_fps['Result'] = 'FP'
    # pd_prediction = pd.concat([pd_tps, pd_fps])
    # pd_prediction.to_csv('./output/bean_prediction_result.csv', sep='\t', index=False)

    # # wine
    # tp, fp, tp_list, fp_list, _ = analyze_ontoloty.get_stats(pd_pairs, match_only=wine_classes)
    # precision = tp / (tp + fp)

    # print('wine precision: {}'.format(precision))
    # print('wine tp/fp: {}/{}'.format(tp, fp))

    # pd_tps = pd.DataFrame(tp_list, columns=columns)
    # pd_tps['Result'] = 'TP'
    # pd_fps = pd.DataFrame(fp_list, columns=columns)
    # pd_fps['Result'] = 'FP'
    # pd_prediction = pd.concat([pd_tps, pd_fps])
    # pd_prediction.to_csv('./output/wine_prediction_result.csv', sep='\t', index=False)

    # # all
    # tp, fp, tp_list, fp_list, _ = analyze_ontoloty.get_stats(pd_pairs)
    # precision = tp / (tp + fp)

    # print('all precision: {}'.format(precision))
    # print('all tp/fp: {}/{}'.format(tp, fp))

    # pd_tps = pd.DataFrame(tp_list, columns=columns)
    # pd_tps['Result'] = 'TP'
    # pd_fps = pd.DataFrame(fp_list, columns=columns)
    # pd_fps['Result'] = 'FP'
    # pd_prediction = pd.concat([pd_tps, pd_fps])
    # pd_prediction.to_csv('./output/all_prediction_result.csv', sep='\t', index=False)


def main():
    """
    Main function.
    """
    set_logging('./output/log/analysis.log')

    # precision, filename = calculate_precision('./data/scores/wiki/pairs_5.pkl')
    # sys.exit()

    alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_mapping_list = [10000]

    # # find best grid search result
    # find_best_grid_search_result(
    #     alpha_list,
    #     num_mapping_list)
    # sys.exit()

    # num_mapping_best = 10000

    # plot_distance_vs_precision_and_TPs()

    # distance_all_models()

    # plot_precision_all_models()

    # plot_precision_with_without_food_product()

    # plot_precision_different_random_seeds()

    # plot_alpha_vs_precision()

    # plot_foodon_analysis()

    # visualize_foodon()

    # visualize_foodon_bean()

    cheese_bean_wine_precision()

if __name__ == '__main__':
    main()
