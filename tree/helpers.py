from collections import Counter
from os.path import join
import os

import pandas as pd
import torch
from common.logger import TqdmToLogger
from tqdm import tqdm
from tree.node import Node
from typing import List, Tuple, Union

class TreeBuilder:
    """
    TreeBuilder class to build tree structure from given files.

    Attributes:
        files: A list of paths to the input files.
        counts: Either a collections.Counter object or a path to the counts file.
        cutoff_level: An integer value for the cutoff level of the tree.
        codes: A list of tuples, each containing a level (integer) and a code (string).
    """

    def __init__(self, files: Union[None, List[str]]=None, counts: Union[None, Counter, str]=None, cutoff_level: int=5):
        """
        Initializes the TreeBuilder with the provided input files, counts, and cutoff level.
        """
        self.files = files or ['data_dumps/sks_dump_diagnose.csv', 'data_dumps/sks_dump_medication.csv']
        self.counts = counts 
        self.cutoff_level = cutoff_level
        self.codes = self.create_levels()

    def build_tree(self) -> Node:
        """
        Builds the tree structure from the input files.
        Returns: The root node of the built tree structure.
        """
        tree = self.create_tree()
        tree.cutoff_at_level(self.cutoff_level)
        tree.extend_leaves(self.cutoff_level)

        counts = self.load_counts()
        tree.base_counts(counts)
        tree.sum_counts()
        tree.redist_counts()

        return tree

    def load_counts(self) -> Union[Counter, torch.Tensor]:
        """
        Loads the counts from the counts object or file.
        Returns: Either a collections.Counter object or a PyTorch tensor of the counts.
        """
        if isinstance(self.counts, Counter):
            return self.counts
        elif isinstance(self.counts, str):
            return torch.load(self.counts)
        else:
            raise TypeError("counts must be a Counter object or a path to the counts file.")

    def create_levels(self) -> List[Tuple[int, str]]:
        """
        Creates the levels of the tree structure from the input files.
        Returns: A list of tuples each containing a level (integer) and a code (string).
        """
        codes = []
        for file in self.files:
            df = pd.read_excel(file)
            level = -1
            prev_code = ''
            for i, (code, text) in df.iterrows():
                if pd.isna(code):
                    level, code = self.handle_nan_code(level, prev_code, text, df, i)
                else:
                    level += len(code) - len(prev_code)
                    prev_code = code 

                if code.startswith('XX'):  # Gets proper code (chapter/topic range)
                    code = text.split()[-1]

                if 'medication' in file:
                    level, code = self.adjust_level_for_medication(level, code)

                codes.append((level, code))

        codes.extend(self.add_background())
        return codes

    def handle_nan_code(self, level: int, prev_code: str, text: str, df: pd.DataFrame, i: int) -> Tuple[int, str]:
        """
        Handles cases where the code is NaN.
        Returns: A tuple containing the updated level and code.
        """
        if text[:3].lower() == 'kap':
            code = 'XX'
        else:
            if pd.isna(df.iloc[i+1].Kode):
                return level, prev_code  # Skip "subsub"-topics (double nans not started by chapter)
            code = 'XXX'

        level += len(code) - len(prev_code)  # Add distance between current and previous code to level
        prev_code = code
        return level, code
    
    def adjust_level_for_medication(self, level: int, code: str) -> Tuple[int, str]:
        """
        Adjusts the level for medication.
        Returns: level, code: The updated level and code.
        """
        if level in [3,4,5]:
            level -= 1
        elif level == 7:
            level -= 2
        return level, code

    def add_background(self)-> List[Tuple[int, str]]:
        """
        Adds background codes to the tree structure.
        Returns: background: A list of background codes.
        """
        return [
            (0, 'BG'), 
            (1, '[GENDER]'), 
            (2, 'BG_Mand'), (2, 'BG_Kvinde'), 
            (1, '[BMI]'), 
            (2, 'BG_underweight'), (2, 'BG_normal'), (2, 'BG_overweight'), 
            (2, 'BG_obese'), (2, 'BG_extremely-obese'), (2, 'BG_morbidly-obese'), (2, 'BG_nan')
        ]

    def create_tree(self)-> Node:
        """
        Creates the tree structure from the levels and codes.
        Returns: root: The root node of the tree.
        """
        root = Node('root')
        parent = root
        for i in range(len(self.codes)):
            level, code = self.codes[i]
            next_level = self.codes[i+1][0] if i < len(self.codes)-1 else level
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

