"""
Authors:
    Simon Kit Sang, Chu - kschu@ucdavis.edu

Description:
    Tokenization maneger for processing the FDC dataset.

To-do:
"""
# standard libraries
import logging as log
import os

# third party libraries
import pandas as pd
import re


class TokenManager:
    """
    Class for tokenizing the FDC data.
    """

    def __init__(self, data_preprocess_dir, replace_csv='FindReplace.csv', cat_csv='simplified_categories.csv'):
        """
        Class initializer.

        Inputs:
            data_preprocess_dir: (str) Directory containing preprocess rules.
            replace_csv: (str) csv containing remove/replace words.
            cat_csv: (str) csv containing category labels
            nutrient_dir: (str) Directory containing FDC nutrient data.
        """
        self.data_preprocess_dir = data_preprocess_dir
        self.replace_csv = os.path.join(self.data_preprocess_dir, replace_csv)
        self.cat_csv = os.path.join(self.data_preprocess_dir, cat_csv)

        # Load FindReplace.csv
        pd_replace = pd.read_csv(self.replace_csv, sep=';')
        pd_replace.drop_duplicates('Find', inplace=True)
        pd_replace['Replace'].fillna('', inplace=True)

        self.pd_replace = pd_replace
        self.brand_list = list(pd_replace[pd_replace['Type'] == 'Brand']['Find'])

        pd_sel = pd_replace[pd_replace['space sep'] == 1]
        self.replace_dict_sep = pd_sel.set_index('Find')['Replace'].to_dict()
        pd_sel = pd_replace[pd_replace['space sep'] != 1]
        self.replace_dict = pd_sel.set_index('Find')['Replace'].to_dict()

        # Load cat_csv
        pd_cat_map = pd.read_csv(self.cat_csv, sep=';')
        self.cat_maps = {}
        for cat_sys in ['branded', 'wweia']:
            df = pd_cat_map[~pd_cat_map[cat_sys].isnull()]
            sys_map = df.set_index(cat_sys)['category'].to_dict()
            self.cat_maps[cat_sys] = sys_map

    def categorize_pd(self, pd_fdc):
        """
        Introduce category field to FDC dataframe.

        Input:
            pd_fdc: (DataFrame) Raw FDC dataframe.
        Ouptut:
            pd_fdc: (DataFrame) FDC dataframe with category column.
        """
        for key in self.cat_maps:
            pd_fdc[key] = pd_fdc[key].fillna('').astype(str)

            cat_map = self.cat_maps[key]
            pd_sel = pd_fdc[pd_fdc[key].isin(cat_map.keys())]
            pd_fdc['category'] = pd_sel[key].map(cat_map)

        pd_fdc['category'].fillna('', inplace=True)

        return pd_fdc

    def pd2brands(self, pd_fdc, brand_size_max=3, filename_brands='brands.csv'):
        """
        Extract brand name from description field of branded fdc data

        Input:
            pd_fdc: (DataFrame) fdc dataframe
        Output:
            brands.csv: file containing all the brands
        """
        pd_branded = pd_fdc[~pd_fdc['branded'].isnull()]

        brands = []
        for description in pd_branded['description']:
            if ',' in description:
                brand = description.split(',')[0]
                if len(brand.split()) <= brand_size_max:
                    brands.append(brand)
        brands = list(set(brands))

        with open(filename_brands, 'w') as output:
            string = '\n'.join(brands)
            output.write(string)

    def find_replace(self, description_raw, keep_brand=False):
        """
        Find and replace/remove in a description text. Originally calibrated
        for description field in FDC data only.

        Warning:
            In the future, each brand must be renamed to non-space-delimited version to make the tokenization.

        Input:
            description: (str) description text of the food
        Output:
            description: (str) replaced/removed description text
        """
        description = description_raw

        if keep_brand:
            keys = [key for key in self.replace_dict.keys() if key not in self.brand_list]
            keys_sep = [key for key in self.replace_dict_sep.keys() if key not in self.brand_list]
        else:
            keys = self.replace_dict.keys()
            keys_sep = self.replace_dict_sep.keys()

        for key in keys:
            description = description.replace(key, self.replace_dict[key])

        for key in keys_sep:
            desc_list = [self.replace_dict_sep[x] if x == key else x for x in description.split()]
            description = ' '.join(desc_list)

        return description

    def tokenize(self, text):
        """
        Tokenize text. Optimized for description only.

        Input:
            text: (str) Text to be tokenized.
        Output:
            tokens: (str) Tokenized text separated by space.
        """
        tokens = text.split()
        tokens = [x.strip(',') for x in tokens]
        tokens = [x.strip('-') for x in tokens]
        tokens = [x.strip('.') for x in tokens]
        tokens = [x.strip() for x in tokens]
        tokens = ' '.join(tokens)

        return tokens

    def clean_ingredient(self, ingredients):
        """
        Clean up ingredient field

        Input:
            ingredient: (str) ingredient field of the food

        Output:
            ingredient: (str) ingredient cleaned
        """
        if ingredients == ingredients:
            ingredients = re.sub('[*().]', '', ingredients.lower())
            ingredients = ' '.join(ingredients.split(','))
            ingredients = ' '.join(ingredients.split())
        else:
            ingredients = ''

        return ingredients

    def remove_numeric(self, description_raw):
        """
        Remove stand-alone numeric from description.
        Stand-alone is splitted by space/tab

        Input:
            description_raw: (str) description text of the food

        Output:
            description: (str) description text of the food without stand-alone numeric
        """
        non_numerics = []
        desc = description_raw.split()

        for i, x in enumerate(desc):
            # check if numeric
            try:
                float(x)

                # save numeric if in front of %
                if i < len(desc) - 1:
                    if desc[i + 1] == '%':
                        non_numerics.append(x)

            except ValueError:
                non_numerics.append(x)

        description = ' '.join(non_numerics)

        return description

    def append_label(self, pd_data, pd_label, columns_match=['fdc_id']):
        """
        Append label to FDC DataFrame

        Input:
            pd_data: (DataFrame) FDC data.
            pd_label: (DataFrame) label(s) (with fdc_id)
            columns_match: (list) list of columns for matching

        Output:
            pd_labelled: (DataFrame) labelled FDC data
        """
        pd_labelled = pd_data.join(pd_label, on=columns_match)
        return pd_labelled
