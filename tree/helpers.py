from collections import Counter
from os.path import join
import os

import pandas as pd
import torch
from common.logger import TqdmToLogger
from data.concept_loader import ConceptLoader
from tqdm import tqdm
from tree.node import Node


class TreeBuilder:
    """
    Build the tree structure from given file.
    Returns:
    A tree structure.
    """
    def __init__(self, file='data_dumps/sks_dump_columns.csv', counts=None, cutoff_level=5):
        self.file = file
        self.counts = counts
        self.cutoff_level = cutoff_level
        self.codes = self.create_levels()

    def build(self):
        tree = self.create_tree()
        tree.add_background()
        tree.cutoff_at_level(self.cutoff_level)
        tree.extend_leaves(self.cutoff_level)
        if self.counts is None:
            self.counts = torch.load('base_counts.pt')
        tree.base_counts(self.counts)
        tree.sum_counts()
        tree.redist_counts()
        return tree

    def create_levels(self):
        """
        Create levels of tree structure.
        Returns:
        A list of tuples with levels and codes.
        """
        with open(self.file, 'r') as f:
            df = pd.read_csv(f)
        codes = []
        level = -1
        prev_code = ''
        for i, (code, text) in df.iterrows():
            level, prev_code = self.update_level_and_code(level, prev_code, code, text)
            codes.append((level, code))
        return codes

    def update_level_and_code(self, level, prev_code, code, text):
        """
        Update level and code based on the given code and text.
        Returns:
        An updated level and code.
        """
        if pd.isna(code):
            code, level = self.handle_nan_code(level, prev_code, text)
        else:
            level += len(code) - len(prev_code)
            prev_code = code 
        return level, prev_code

    def handle_nan_code(self, level, prev_code, text):
        """
        Handle cases where the code is NaN.
        Returns:
        The updated code and level.
        """
        if text[:3].lower() == 'kap':
            code = 'XX'
            level += 2 - len(prev_code)
        else:
            code = 'XXX'
            level += 3 - len(prev_code)
        prev_code = code 
        code = text.split()[-1]
        return code, level

    def create_tree(self):
        root = Node('root')
        parent = root
        for i in range(len(self.codes)):
            level, code = self.codes[i]
            next_level = self.codes[i+1][0] if i < len(self.codes)-1 else level
            dist = next_level - level 
            parent, dist = self.update_parent_and_dist(parent, dist, code)
        return root 

    def update_parent_and_dist(self, parent, dist, code):
        if dist >= 1:
            for _ in range(dist):
                parent.add_child(code)
                parent = parent.children[-1]
        elif dist <= 0:
            parent.add_child(code)
            for _ in range(0, dist, -1):
                parent = parent.parent
        return parent, dist


def get_counts(cfg, logger):
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

