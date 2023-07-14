import os
from data.dataset import (HierarchicalDataset, HierarchicalLargeDataset,
                          MLMDataset, MLMLargeDataset)


def create_datasets(cfg, hierarchical:bool=False, file_prefix:str='tokenized_train'):
    """
    This function is used to create datasets based on the configuration provided.
    Args:
        cfg (Config): Configuration object containing paths, dataset information etc.
        hierarchical (bool, optional): If True, create hierarchical datasets. Otherwise, create MLM datasets. Defaults to False.
        file_prefix (str, optional): Prefix for files to be matched. Defaults to 'tokenized_train'.
    Returns:
        tuple: Returns a tuple containing train_dataset and val_dataset.
    """
    data_path = cfg.paths.data_path
    train_files = [file for file in os.listdir(data_path) if file.startswith(file_prefix)]

    DatasetClass = None

    if hierarchical:
        DatasetClass = HierarchicalLargeDataset if len(train_files) > 1 else HierarchicalDataset
    else:
        DatasetClass = MLMLargeDataset if len(train_files) > 1 else MLMDataset

    train_dataset, val_dataset = load_datasets(cfg, DatasetClass)
    return train_dataset, val_dataset

def load_datasets(cfg, DS):
    """
    This function is used to load datasets based on the given DatasetClass and configuration.
    Args:
        cfg (Config): Configuration object containing paths, dataset information etc.
        DS (class): The class of the dataset to be created. It can be either of MLMLargeDataset, MLMDataset, 
                    HierarchicalLargeDataset, or HierarchicalDataset.
    Returns:
        tuple: Returns a tuple containing train_dataset and val_dataset.
    """
    data_path = cfg.paths.data_path
    train_dataset = DS(data_path, 'train', **cfg.dataset)
    val_dataset = DS(data_path, 'val', **cfg.dataset)
    return train_dataset, val_dataset
