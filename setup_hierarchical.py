import os
from os.path import join

import torch
from common.config import load_config
from common.setup import prepare_directory_hierarchical
from tree.helpers import build_tree, get_counts

# from azureml.core import Dataset
# from azure_run.run import Run
# from azure_run import datastore

# run = Run
# run.name(f"Pretrain base on med/diag")

# dataset = Dataset.File.from_files(path=(datastore(), 'PHAIR/formatted_data/diagnosis_medication'))

config_path = join('configs', 'h_setup.yaml')

def setup_hierarchical(config_path=config_path):
    cfg = load_config(config_path)
    logger = prepare_directory_hierarchical(config_path, cfg)  
    logger.info('Mount Dataset')
    # mount_context = dataset.mount()
    # mount_context.start()  # this will mount the file streams
    
    # cfg.paths.features = mount_context.mount_point
    
    data_dir = cfg.paths.features
    
    logger.info('Get Counts')
    counts = get_counts(cfg, logger=logger)
    logger.info('Build Tree')
    tree = build_tree(counts=counts, cutoff_level=cfg.cutoff_level)
    logger.info('Create Vocabulary')
    vocabulary = tree.create_vocabulary()

    torch.save(vocabulary, join(data_dir,'hierarchical', 'vocabulary.pt'))
    torch.save(counts, join(data_dir, 'hierarchical', 'base_counts.pt'))
    torch.save(tree, join(data_dir, 'hierarchical','tree.pt'))
    # mount_context.stop()

if __name__ == '__main__':
    setup_hierarchical(config_path=config_path)