from os.path import join

import torch
from common.config import load_config
from common.setup import prepare_directory
from tree.helpers import TreeBuilder, get_counts

# from azureml.core import Dataset
# from azure_run.run import Run
# from azure_run import datastore

# run = Run
# run.name(f"Pretrain base on med/diag")

# dataset = Dataset.File.from_files(path=(datastore(), 'PHAIR/formatted_data/diagnosis_medication'))

config_path = join('configs', 'h_data.yaml')

def setup_hierarchical(config_path=config_path):
    cfg = load_config(config_path)
    logger = prepare_directory(config_path, cfg)  
    logger.info('Mount Dataset')
    # mount_context = dataset.mount()
    # mount_context.start()  # this will mount the file streams
    
    # cfg.loader.data_dir = mount_context.mount_point
    
    logger.info('Get Counts')
    counts = get_counts(cfg)
    logger.info('Build Tree')
    tree_builder = TreeBuilder(counts=counts)
    tree = tree_builder.build()
    logger.info('Create Vocabulary')
    vocabulary = tree.create_vocabulary()

    torch.save(vocabulary, join(cfg.output_dir, 'vocabulary.pt'))
    torch.save(counts, join(cfg.output_dir, 'base_counts.pt'))
    return tree, vocabulary

if __name__ == '__main__':
    print("Preparing hierarchical data...")
    setup_hierarchical(config_path=config_path)