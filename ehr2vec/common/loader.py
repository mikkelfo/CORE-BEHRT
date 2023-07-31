import glob
import os
from os.path import join

from data.dataset import HierarchicalMLMDataset, MLMDataset


def create_datasets(cfg, hierarchical:bool=False):
    """
    This function is used to create datasets based on the configuration provided.
    """
    if hierarchical:
        DatasetClass = HierarchicalMLMDataset
    else:
        DatasetClass = MLMDataset
    train_dataset, val_dataset = load_datasets(cfg, DatasetClass)
    return train_dataset, val_dataset

def load_datasets(cfg, DS):
    """
    This function is used to load datasets based on the given DatasetClass and configuration.
    """
    data_path = cfg.paths.data_path
    train_dataset = DS(data_path, 'train', **cfg.dataset, **cfg.train_data)
    val_dataset = DS(data_path, 'val', **cfg.dataset, **cfg.val_data)
    return train_dataset, val_dataset

def check_directory_for_features(dir_, logger):
    features_dir = join(dir_, 'features')
    if os.path.exists(features_dir):
        if len(glob.glob(join(features_dir, 'features_*.pt')))>0:
            logger.warning(f"Features already exist in {features_dir}.")
            logger.warning(f"Skipping feature creation.")
        return True
    else:
        return False