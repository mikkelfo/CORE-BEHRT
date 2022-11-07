import torch


def load_features(train_file="featurxes.train", test_file="features.test"):
    with open(train_file) as f:
        train_set = torch.load(f)
    with open(test_file) as f:
        test_set = torch.load(f)

    return train_set, test_set

def load_vocabulary(file="vocabulary.pt"):
    with open(file) as f:
        vocabulary = torch.load(f)
    return vocabulary

