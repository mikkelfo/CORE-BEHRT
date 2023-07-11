from data.dataset import MLMLargeDataset
from data.dataset import HierarchicalLargeDataset
from os.path import join
import torch

def create_datasets(cfg):
    data_path = cfg.paths.data_path
    train_dataset = MLMLargeDataset(data_path, 'train', **cfg.dataset)
    val_dataset = MLMLargeDataset(data_path, 'val', **cfg.dataset)
    vocabulary = train_dataset.vocabulary
    return train_dataset, val_dataset, vocabulary

def create_hierarchical_dataset(cfg):
    data_path = cfg.paths.data_path
    hierarchical_path = join(data_path, 'hierarchical')
    tree = torch.load(join(hierarchical_path, 'tree.pt'))
    train_dataset = HierarchicalLargeDataset(data_path, 'train', tree=tree, **cfg.dataset)
    val_dataset = HierarchicalLargeDataset(data_path, 'val', tree=tree, **cfg.dataset)
    return train_dataset, val_dataset