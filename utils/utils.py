import torch


def load_features(train_file="dataset.train", test_file="dataset.test"):
    with open(train_file, 'rb') as f:
        train_set = torch.load(f)
    with open(test_file, 'rb') as f:
        test_set = torch.load(f)

    return train_set, test_set

def load_vocabulary(file="vocabulary.pt"):
    with open(file, 'rb') as f:
        vocabulary = torch.load(f)
    return vocabulary

def to_device(*tensors, device):
    for tensor in tensors:
        tensor.to(device)

