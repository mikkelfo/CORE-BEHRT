"""Script to prepare hierarchical data. config template: h_setup.yaml. main_pretrain.py needs to be run first. Here, we use the tokenized data to build the hierarchical vocabulary and tree and the hierarchical target."""
import os
from os.path import join

import torch
from ehr2vec.common.azure import AzurePathContext, save_to_blobstore
from ehr2vec.common.config import load_config
from ehr2vec.common.setup import DirectoryPreparer, get_args
from ehr2vec.tree.tree import TreeBuilder, get_counts

BLOBSTORE = 'PHAIR'
CONFIG_NAME = 'h_setup.yaml'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

def setup_hierarchical(config_path=config_path):
    cfg = load_config(config_path)
    data_dir = cfg.paths.features
    hierarchical_name = cfg.paths.get('hierarchical_name', 'hierarchical')

    cfg, _, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_hierarchical_setup()
    logger = DirectoryPreparer(config_path).prepare_directory_hierarchical(cfg.paths.output_path, hierarchical_name)  
    
    logger.info('Get Counts')
    counts = get_counts(cfg, logger=logger)
    logger.info('Build Tree')
    tree = TreeBuilder(counts=counts, **cfg.tree).build()
    logger.info('Create Vocabulary')
    vocabulary = tree.create_vocabulary()
    logger.info('Save hierarchical vocabulary')
    hierarchical_path = join(cfg.paths.output_path, hierarchical_name)
    torch.save(vocabulary, join(hierarchical_path, 'vocabulary.pt'))
    logger.info('Save base counts')
    torch.save(counts, join(hierarchical_path, 'base_counts.pt'))
    logger.info('Save tree')
    torch.save(tree, join(hierarchical_path,'tree.pt'))
    logger.info('Construct and Save tree matrix')
    torch.save(tree.get_tree_matrix(), join(hierarchical_path, 'tree_matrix.pt'))
    if cfg.env == 'azure':
        save_to_blobstore(local_path=hierarchical_name, remote_path=join(BLOBSTORE, data_dir, hierarchical_name))
        mount_context.stop()

if __name__ == '__main__':
    setup_hierarchical(config_path=config_path)