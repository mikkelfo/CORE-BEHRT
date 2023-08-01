"""Script to prepare hierarchical data. config template: h_setup.yaml . main_pretrain.py needs to be run first. Here, we use the tokenized data to build the hierarchical vocabulary and tree and the hierarchical target."""
import os
from os.path import join

import torch
from common import azure
from common.config import load_config
from common.setup import prepare_directory_hierarchical
from tree.tree import TreeBuilder, get_counts

config_path = join('configs', 'h_setup.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

def setup_hierarchical(config_path=config_path):
    cfg = load_config(config_path)
    data_dir = cfg.paths.features
    if cfg.env=='azure':
        _, mount_context = azure.setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_dir
        cfg.paths.features = join(mount_dir, cfg.paths.features)
        data_dir = "outputs/data"
    logger = prepare_directory_hierarchical(config_path, cfg)  
    
    
    
    logger.info('Get Counts')
    counts = get_counts(cfg, logger=logger)
    logger.info('Build Tree')
    tree = TreeBuilder(counts=counts, cutoff_level=cfg.cutoff_level).build()
    logger.info('Create Vocabulary')
    vocabulary = tree.create_vocabulary()
    logger.info('Save hierarchical vocabulary')
    torch.save(vocabulary, join(data_dir,'hierarchical', 'vocabulary.pt'))
    logger.info('Save base counts')
    torch.save(counts, join(data_dir, 'hierarchical', 'base_counts.pt'))
    logger.info('Save tree')
    torch.save(tree, join(data_dir, 'hierarchical','tree.pt'))
    logger.info('Construct and Save tree matrix')
    torch.save(tree.get_tree_matrix(), join(data_dir, 'hierarchical', 'tree_matrix.pt'))
    if cfg.env == 'azure':
        mount_context.stop()

if __name__ == '__main__':
    setup_hierarchical(config_path=config_path)