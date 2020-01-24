"""
Authors:
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Preprocess the FDC data.

To-do:
"""
# standard imports
import os
import random
import numpy as np

# third party imports
import pandas as pd
import gensim
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from sklearn.manifold import TSNE


def main():
    """
    Main function.
    """

    # settings
    preprocess_dir = '/home/jyoun/Jason/Research/FoodOntology/output'
    filename_token = os.path.join(preprocess_dir, 'tokenized.csv')
    flag_ingredients = True
    len_ingredients = 3  # take the whole ingredient if -1

    flag_cat_filter = True
    cat_filter = ['Meat']

    model_dir = '/home/jyoun/Jason/Research/FoodOntology/data/word2vec'
    filename_model = os.path.join(model_dir, 'modelfile')
    flag_retrain = True

    # model hyperparameters
    hidden_layer_size = 24
    window_size = 100
    min_count = 10
    workers_num = 4
    epochs = 100

    # tSNE parameters
    perplexity = 30
    n_iter = 1000
    random_state = 1  # fix random seed to ensure reproducibility of plots

    # load tokens
    pd_token = pd.read_csv(filename_token, sep=';')
    pd_token[['description', 'ingredients']] = pd_token[['description', 'ingredients']].fillna('')
    pd_token['category'] = pd_token['category'].fillna('Miscellaneous')

    # filter to simplified categories
    if flag_cat_filter:
        pd_token = pd_token[pd_token['category'].isin(cat_filter)]

    if flag_ingredients:
        if len_ingredients == -1:
            pd_token['token'] = pd_token['description'] + pd_token['ingredients']
        else:
            pd_token['ingredients filtered'] = pd_token['ingredients'].apply(lambda ingredients: ' '.join([x for x in ingredients.split()][:len_ingredients]))
            pd_token['token'] = pd_token['description'] + pd_token['ingredients filtered']
    else:
        pd_token['token'] = pd_token['description']
    pd_token = pd_token[pd_token['token'] != '']

    token_list = []
    for tokens in pd_token['token']:
        token_list.append(tokens.split())

    # train word2vec
    if os.path.exists(filename_model) and not flag_retrain:
        word2vec = gensim.models.Word2Vec.load(filename_model)
    else:
        word2vec = gensim.models.Word2Vec(
            token_list,
            size=hidden_layer_size,
            window=window_size,
            min_count=min_count,
            workers=workers_num)

        word2vec.train(
            token_list,
            total_examples=len(token_list),
            epochs=epochs)

        word2vec.save(filename_model)

    # generate coordinate for food (not individual token)
    food_coor_dict = {}
    for food in pd_token['token']:
        tokens = food.split()
        food_coor = np.zeros((len(tokens), hidden_layer_size))

        for i, token in enumerate(tokens):
            try:
                food_coor[i] = word2vec[token]
            except KeyError:
                pass

        food_coor = np.sum(food_coor, axis=0)
        food_coor_dict[food] = food_coor

    # tSNE for foods
    tsne_food = TSNE(perplexity=perplexity, n_components=2, n_iter=n_iter, random_state=random_state)
    tsne_food.fit(np.asarray([x for x in food_coor_dict.values()]))

    food_tsne_dict = {}
    for food in food_coor_dict.keys():
        food_tsne = tsne_food.fit_transform(food_coor_dict[food].reshape(-1, 1))
        food_tsne_dict[food] = food_tsne.reshape(-1)


if __name__ == '__main__':
    main()
