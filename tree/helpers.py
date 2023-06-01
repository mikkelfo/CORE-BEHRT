import pandas as pd
from tree.tree import Node
import torch


def setup_hierarchical():
    counts = get_counts()
    tree = build_tree(counts=counts)
    vocabulary = tree.create_vocabulary()

    torch.save(vocabulary, 'vocabulary.pt')
    torch.save(counts, 'base_counts.pt')
    return tree, vocabulary

# TODO: Implement
def get_counts():
    return {}


def build_tree(file='data_dumps/sks_dump_columns.xlsx', counts=None, cutoff_level=5):
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
    df = pd.read_excel(file)

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


def flatten(data):
    def _flatten(data):
        for element in data:
            if isinstance(element, list):
                yield from _flatten(element)
            else:
                yield element
    return list(_flatten(data))


if __name__ == '__main__':
    print("Preparing hierarchical data...")
    setup_hierarchical()