import pandas as pd
import torch
from common.logger import TqdmToLogger
from data.concept_loader import ConceptLoader
from tqdm import tqdm
from tree.node import Node
from collections import Counter


def get_counts(cfg, logger):
    """
    Get the counts of unique values from data loaded with the ConceptLoader.
    Returns:
    A dictionary with counts of unique values.
    """
    conceptloader = ConceptLoader(**cfg.loader)
    all_data_counter = Counter()

    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Batch Process Data', file=TqdmToLogger(logger))):
        codes = concept_batch.CONCEPT
        info = [patient_batch[col] for col in cfg.features.background]
        all_data_batch = pd.concat([codes, *info])
        all_data_counter.update(all_data_batch.value_counts().to_dict())

    return all_data_counter

def build_tree(file='data_dumps/sks_dump_columns.csv', counts=None, cutoff_level=5):
    codes = create_levels(file)
    tree = create_tree(codes)
    tree.add_background()
    tree.cutoff_at_level(cutoff_level)
    tree.extend_leaves(cutoff_level)

    if counts is None:
        counts = torch.load('base_counts.pt')
    tree.base_counts(counts)
    tree.sum_counts()
    tree.redist_counts()
    return tree

def create_levels(file='data_dumps/sks_dump_columns.xlsx'):
    # df = pd.read_excel(file)
    df = pd.read_csv(file)
    codes = []
    level = -1
    prev_code = ''
    for i, (code, text) in df.iterrows():
        if pd.isna(code):
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

        codes.append((level, code))

    return codes

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

