import torch
from tree.helpers import get_counts, build_tree


def setup_hierarchical():
    counts = get_counts()
    tree = build_tree(counts=counts)
    vocabulary = tree.create_vocabulary()

    torch.save(vocabulary, "data/processed/vocabulary.pt")
    torch.save(counts, "data/extra/base_counts.pt")
    return tree, vocabulary


if __name__ == "__main__":
    print("Preparing hierarchical data...")
    setup_hierarchical()
