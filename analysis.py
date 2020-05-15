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

# third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

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

    tp, fp, _, _ = analyze_ontoloty.get_stats(pd_pairs)
    precision = tp / (tp + fp)

    return (precision, file)


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


def plot_alpha_vs_precision(
        num_mapping_best,
        alpha_list,
        analyze_ontoloty):

    pd_grid_search_result = pd.read_csv('./output/grid_search_result.txt', sep='\t')

    precision_list = []
    for alpha in alpha_list:
        filename = 'pairs_alpha{}_N{}.pkl'.format(alpha, num_mapping_best)
        pd_match = pd_grid_search_result[
            pd_grid_search_result['Filename'].str.contains(filename)]

        assert pd_match.shape[0] == 1

        precision_list.append(pd_match['Precision'].tolist()[0])

    fig = plt.figure()
    plt.plot(alpha_list, precision_list)
    plt.grid(True)
    # plt.axis([0, iterations[-1], 0, 1.0])
    plt.xlabel('alpha')
    plt.ylabel('Precision')
    # plt.title('Best model ({}) PR curve'.format(best_classifier))
    # plt.legend(lines, labels, loc='upper right', prop={'size': 8})

    save_figure(fig, './output/alpha_vs_precision.png')


def plot_distance_vs_precision_and_TPs(
        alpha_best,
        num_mapping_best,
        analyze_ontoloty):
    filename = './data/scores/pairs_alpha{}_N{}.pkl'.format(alpha_best, num_mapping_best)
    population_pairs = load_pkl(filename)
    iterations = list(population_pairs.keys())
    iterations.sort()

    # distance vs TPs
    pairs = []
    for iteration in iterations:
        pairs.extend(population_pairs[iteration])
    pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
    pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
    pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

    _, _, _, distance_distribution = analyze_ontoloty.get_stats(pd_pairs)

    # distance vs precision
    precision_list = []
    for allow_distance in range(max(distance_distribution)+1):
        log.info('Processing allowed distance: %d', allow_distance)

        pairs = []
        for iteration in iterations:
            pairs.extend(population_pairs[iteration])
        pd_pairs = pd.DataFrame(pairs, columns=['Parent', 'Child'])
        pd_pairs = pd_pairs[~pd_pairs['Parent'].isin(classes_without_embedding)]
        pd_pairs = pd_pairs[~pd_pairs['Child'].isin(entities_without_embedding)]

        tp, fp, _, _ = analyze_ontoloty.get_stats(pd_pairs, allow_distance=allow_distance)
        precision_list.append(tp / (tp + fp))
        print(precision_list)

    # plot
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.hist(
        distance_distribution,
        bins=range(max(distance_distribution)+2),
        color='teal',
        edgecolor='black',
        linewidth=1)
    ax2.plot([i+0.5 for i in range(max(distance_distribution)+1)], precision_list, color='k')

    ax1.set_xlabel('Distance')
    ax1.set_ylabel('# of TPs', color='teal')
    ax2.set_ylabel('Precision', color='k')
    plt.grid(True)
    plt.xticks([i+0.5 for i in range(max(distance_distribution)+1)], range(max(distance_distribution)+1))

    save_figure(fig, './output/distance_vs_precision_and_tps.png')


def plot_foodon_analysis():
    # pie chart fig 4
    fig, ax = plt.subplots()

    size = 0.3
    vals = np.array([[331., 2433.], [3111., 7754.]])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

    labels_outer = [
        'Classes\n(2,764)',
        'Entities\n(10,865)']

    ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'), labels=labels_outer)

    labels_inner = [
        'Non-candidate class\n(331)',
        'Candidate class\n(2,433)',
        'Seed entity\n(3,111)',
        'Candidate entity\n(7,754)']

    ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'), labels=labels_inner, labeldistance=0.5)

    ax.set(aspect='equal')
    save_figure(fig, './output/foodon_class_entity_pie_chart.png')

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

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(level_1_dict_sorted.values(), wedgeprops=dict(width=0.5), startangle=-40)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=None, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(list(level_1_dict_sorted.keys())[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    save_figure(fig, './output/foodon_level_1_pie_chart.png')

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
    save_figure(fig, './output/foodon_number_of_paths_vs_classes.png')

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
    save_figure(fig, './output/foodon_num_entities_vs_num_classes.png')

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
    save_figure(fig, './output/foodon_level_vs_num_classes.png')


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
    save_figure(fig, './output/foodon_visualization.png')


def main():
    """
    Main function.
    """
    set_logging()

    # ret_val = calculate_precision('./data/scores/pairs_lcsseq.pkl')
    # print(ret_val)
    # sys.exit()

    # alpha_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_list = [0.8]
    num_mapping_list = [7748]

    # find best grid search result
    find_best_grid_search_result(
        alpha_list,
        num_mapping_list)

    sys.exit()

    alpha_best = 0.8
    num_mapping_best = 7748
    analyze_ontoloty = AnalyzeOntology('./config/analyze_ontology.ini')

    # plot_alpha_vs_precision(
    #     num_mapping_best,
    #     alpha_list,
    #     analyze_ontoloty)

    # plot_distance_vs_precision_and_TPs(
    #     alpha_best,
    #     num_mapping_best,
    #     analyze_ontoloty)

    plot_foodon_analysis()

    # visualize_foodon()


if __name__ == '__main__':
    main()
