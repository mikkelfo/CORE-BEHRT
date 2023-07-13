from data.dataset import MLMLargeDataset, MLMDataset
from data.dataset import HierarchicalLargeDataset, HierarchicalDataset
from os.path import join
import torch

def create_datasets(cfg):
    data_path = cfg.paths.data_path
    if cfg.large_dataset:
        train_dataset = MLMLargeDataset(data_path, 'train', **cfg.dataset)
        val_dataset = MLMLargeDataset(data_path, 'val', **cfg.dataset)
    else:
        train_encoded = torch.load(join(data_path, 'tokenized','train_encoded.pt'))
        val_encoded = torch.load(join(data_path, 'tokenized','val_encoded.pt'))
        vocabulary = torch.load(join(data_path, 'vocabulary.pt'))
        train_dataset = MLMDataset(train_encoded, vocabulary=vocabulary, **cfg.dataset)
        val_dataset = MLMDataset(val_encoded, vocabulary=vocabulary, **cfg.dataset)
        
    return train_dataset, val_dataset, vocabulary

def create_hierarchical_dataset(cfg):
    data_path = cfg.paths.data_path
    hierarchical_path = join(data_path, 'hierarchical')
    tree = torch.load(join(hierarchical_path, 'tree.pt'))
    if cfg.large_dataset:
        train_dataset = HierarchicalLargeDataset(data_path, 'train', tree=tree, **cfg.dataset)
        val_dataset = HierarchicalLargeDataset(data_path, 'val', tree=tree, **cfg.dataset)
    else:
        train_encoded = torch.load(join(data_path, 'tokenized','train_encoded.pt'))
        val_encoded = torch.load(join(data_path, 'tokenized','val_encoded.pt'))
        vocabulary = torch.load(join(data_path, 'vocabulary.pt'))
        train_dataset = HierarchicalDataset(train_encoded, vocabulary=vocabulary, **cfg.dataset)
        val_dataset = HierarchicalDataset(val_encoded, vocabulary=vocabulary, **cfg.dataset)
    return train_dataset, val_dataset