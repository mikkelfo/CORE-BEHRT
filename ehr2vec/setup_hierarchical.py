"""Script to prepare hierarchical data. config template: h_setup.yaml . main_pretrain.py needs to be run first. Here, we use the tokenized data to build the hierarchical vocabulary and tree and the hierarchical target."""
import os
from os.path import join

import torch
from common.azure import setup_azure
from common.config import load_config
from common.setup import prepare_directory_hierarchical, get_args
from tree.tree import TreeBuilder, get_counts

args = get_args("h_setup.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def setup_hierarchical(config_path=config_path):
    cfg = load_config(config_path)
    data_dir = cfg.paths.features
    hierarchical_name = cfg.paths.get('hierarchical_name', 'hierarchical')
    if cfg.env=='azure':
        _, mount_context = setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_point
        cfg.paths.features = join(mount_dir, cfg.paths.features)
        data_dir = "outputs/data"
    logger = prepare_directory_hierarchical(config_path, data_dir, hierarchical_name)  
    
    logger.info('Get Counts')
    counts = get_counts(cfg, logger=logger)
    logger.info('Build Tree')
    tree = TreeBuilder(counts=counts, **cfg.tree).build()
    logger.info('Create Vocabulary')
    vocabulary = tree.create_vocabulary()
    logger.info('Save hierarchical vocabulary')
    hierarchical_path = join(data_dir, hierarchical_name)
    torch.save(vocabulary, join(hierarchical_path, 'vocabulary.pt'))
    logger.info('Save base counts')
    torch.save(counts, join(hierarchical_path, 'base_counts.pt'))
    logger.info('Save tree')
    torch.save(tree, join(hierarchical_path,'tree.pt'))
    logger.info('Construct and Save tree matrix')
    torch.save(tree.get_tree_matrix(), join(hierarchical_path, 'tree_matrix.pt'))
    if cfg.env == 'azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join('outputs', 'data'), datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", cfg.paths.features))
        mount_context.stop()

if __name__ == '__main__':
    setup_hierarchical(config_path=config_path)