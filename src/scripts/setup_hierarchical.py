import os
import hydra
import torch
from src.tree.helpers import get_counts, build_tree


@hydra.main(config_path="../../configs/data", config_name="hierarchical")
def setup_hierarchical(cfg):
    counts = get_counts(cfg)
    tree = build_tree(cfg, counts=counts)
    vocabulary = tree.create_vocabulary()

    torch.save(vocabulary, os.path.join(cfg.paths.data_dir, cfg.paths.vocabulary))
    torch.save(counts, os.path.join(cfg.paths.extra_dir, cfg.paths.base_counts))
    torch.save(tree, os.path.join(cfg.paths.extra_dir, cfg.paths.tree))

    return tree, vocabulary


if __name__ == "__main__":
    print("Preparing hierarchical data...")
    setup_hierarchical()
