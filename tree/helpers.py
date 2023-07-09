from collections import Counter
from os.path import join
import os

import pandas as pd
import torch
from common.logger import TqdmToLogger
from tqdm import tqdm
from tree.node import Node


class TreeBuilder:
    def __init__(self, files, counts, cutoff_level=None):
        self.files = files
        self.counts = counts
        self.cutoff_level = cutoff_level

    def build(self):
        tree_codes = self.create_tree_codes()
        tree = self.create_tree(tree_codes)
        tree.cutoff_at_level(self.cutoff_level)
        tree.extend_leaves(self.cutoff_level)
        tree.base_counts()
        tree.sum_counts()
        tree.redist_counts()

        return tree

    def create_tree_codes(self):
        codes = []
        for file in self.files:
            data_codes = self.get_codes_from_data(file)
            
            df = pd.read_csv(file)
            df = self.augment_database(df, data_codes)
            level = -1
            prev_code = ''
            for i, (code, text) in df.iterrows():
                if pd.isna(code):   # Only for diagnosis
                    # Manually set nan codes for Chapter and Topic (as they have ranges)
                    if text[:3].lower() == 'kap':
                        code = 'XX'             # Sets Chapter as level 2 (XX)
                    else:
                        if pd.isna(df.iloc[i+1].Kode):  # Skip "subsub"-topics (double nans not started by chapter)
                            continue
                        code = 'XXX'            # Sets Topic as level 3 (XXX)

                level += len(code) - len(prev_code)  # Add distance between current and previous code to level
                prev_code = code                # Set current code as previous code

                if code.startswith('XX'):       # Gets proper code (chapter/topic range)
                    code = text.split()[-1]

                # Needed to fix the levels for medication
                if 'medication' in file and level in [3,4,5]:
                    codes.append((level-1, code))
                elif 'medication' in file and level == 7:
                    codes.append((level-2, code))
                else:
                    codes.append((level, code))

        # Add background
        background = [
            (0, 'BG'), 
                (1, '[GENDER]'), 
                    (2, 'BG_Mand'), (2, 'BG_Kvinde'), (2, 'BG_nan'), (2, 'BG_F'), (2, 'BG_M'),
                (1, '[BMI]'), 
                    (2, 'BG_underweight'), (2, 'BG_normal'), (2, 'BG_overweight'), (2, 'BG_obese'), (2, 'BG_extremely-obese'), (2, 'BG_morbidly-obese'), (2, 'BG_nan')
            ]
        background = self.augment_background(background)
        codes.extend(background)

        return codes
    
    @staticmethod
    def create_tree(codes):
        root = Node('root')
        parent = root
        for i in range(len(codes)):
            level, code = codes[i]
            next_level = codes[i+1][0] if i < len(codes)-1 else level
            dist = next_level - level 

            if dist >= 1:
                for _ in range(dist):
                    parent.add_child(code)
                    parent = parent.children[-1]
            elif dist <= 0:
                parent.add_child(code)
                for _ in range(0, dist, -1):
                    parent = parent.parent
        return root  

    def augment_background(self, background:list)->list:
        """Takes a list of background codes and returns a list of background codes with counts."""
        background_codes = [k for k in self.counts.keys() if k.startswith('BG')]
        for code in background_codes:
            code_ls = code.split('_')
            
            if len(code_ls)==2:
                type_ = (1, '[EXTRA]')
                if type_ not in background:
                    background.append(type_)
            elif len(code_ls)==3:
                type_ = (1, '['+code_ls[1]+']')
                if type_ not in background:
                    background.append(type_)
            else:
                raise NotImplementedError(f'Background code {code} does not follow standard format BG_Type_Value or BG_Value')
            insert_index = background.index(type_)+1
            background.insert(insert_index, (2, code))
        return background

    def get_codes_from_data(self, file:str)->dict:
        if 'diagnose' in file:
            return {code: count for code, count in self.counts.items() if code.startswith('D')}
        elif 'medication' in file:
            return {code: count for code, count in self.counts.items() if code.startswith('M')}
        else:
            raise NotImplementedError

    def augment_database(df:pd.DataFrame, data_codes:dict)->pd.DataFrame:
        """Takes a DataFrame and a dictionary of codes and returns a DataFrame with the codes inserted in the correct position."""
        df_data = pd.DataFrame(list(data_codes.items()), columns=['Kode', 'Tekst'])
        # Iterate over the rows of the new DataFrame
        for idx, row in df_data.iterrows():
            # Find the correct position in athe original DataFrame where the new row should be inserted
            insert_position = df.index[df['Kode'] > row['Kode']].min()
            # If there is no such position, append the row at the end
            if pd.isna(insert_position):
                df = df.append(row)
            else:
                # Insert the new row at this position in the original DataFrame
                df = pd.concat([df.loc[:insert_position - 1], pd.DataFrame(row).T, df.loc[insert_position:]]).reset_index(drop=True)

        # Reset the index of the DataFrame
        df = df.reset_index(drop=True, inplace=False)
        return df


def get_counts(cfg, logger)-> dict:
    """Takes a cfg and logger and returns a dictionary of counts for each code in the vocabulary."""
    data_path = cfg.paths.features
    vocabulary = torch.load(join(data_path, 'vocabulary.pt'))
    inv_vocab = {v: k for k, v in vocabulary.items()}

    train_val_files = [
        join(data_path, 'tokenized', f) 
        for f in os.listdir(join(data_path, 'tokenized')) 
        if f.startswith(('tokenized_train', 'tokenized_val'))
    ]
    counts = Counter()
    for f in tqdm(train_val_files, desc="Count" ,file=TqdmToLogger(logger)):
        tokenized_features = torch.load(f)
        counts.update(inv_vocab[code] for codes in tokenized_features['concept'] for code in codes)

    return dict(counts)

