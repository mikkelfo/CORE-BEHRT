import torch


def load_features(train_file="features.train", test_file="features.test"):
    with open(train_file) as f:
        train_codes, train_segments = torch.load(f)
    with open(test_file) as f:
        test_codes, test_segments = torch.load(f)

    return  train_codes, train_segments, test_codes, test_segments

def load_vocabulary(file="vocabulary.pt"):
    with open('vocabulary.pt') as f:
        vocabulary = torch.load(f)
    return vocabulary