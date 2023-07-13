from os.path import join

from data.dataset import (HierarchicalDataset, HierarchicalLargeDataset,
                          MLMDataset, MLMLargeDataset)


def create_datasets(cfg):
    if cfg.large_dataset:
        train_dataset, val_dataset = load_datasets(cfg, MLMLargeDataset)
    else:
        train_dataset, val_dataset = load_datasets(cfg, MLMDataset)
        
    return train_dataset, val_dataset

def create_hierarchical_dataset(cfg):
    if cfg.large_dataset:
        train_dataset, val_dataset = load_datasets(cfg, HierarchicalLargeDataset)
    else:
        train_dataset, val_dataset = load_datasets(cfg, HierarchicalDataset)
    return train_dataset, val_dataset

def load_datasets(cfg, DS):
    data_path = cfg.paths.data_path
    train_dataset = DS(data_path, 'train', **cfg.dataset)
    val_dataset = DS(data_path, 'val', **cfg.dataset)
    return train_dataset, val_dataset
