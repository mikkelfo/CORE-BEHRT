from data.dataset import MLMLargeDataset
from data.dataset import HierarchicalLargeDataset

def create_datasets(cfg):
    data_path = cfg.paths.data_path
    train_dataset = MLMLargeDataset(data_path, 'train', **cfg.dataset)
    val_dataset = MLMLargeDataset(data_path, 'val', **cfg.dataset)
    vocabulary = train_dataset.vocabulary
    return train_dataset, val_dataset, vocabulary

def create_hierarchical_dataset(cfg):
    data_path = cfg.paths.data_path
    train_dataset = HierarchicalLargeDataset(data_path, 'train', **cfg.dataset)
    val_dataset = HierarchicalLargeDataset(data_path, 'val', **cfg.dataset)
    vocabulary = train_dataset.vocabulary
    return train_dataset, val_dataset, vocabulary