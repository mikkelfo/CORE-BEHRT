from data.dataset import MLMLargeDataset


def create_datasets(cfg):
    data_path = cfg.paths.data_path
    train_dataset = MLMLargeDataset(data_path, 'train', **cfg.dataset)
    val_dataset = MLMLargeDataset(data_path, 'val', **cfg.dataset)
    vocabulary = train_dataset.vocabulary
    return train_dataset, val_dataset, vocabulary