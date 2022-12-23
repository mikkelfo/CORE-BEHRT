import torch
import numpy as np


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

def extract_age(row, age_dict):
    if row['Key.Patient'] in age_dict:
        timestamp, age = age_dict[row['Key.Patient']]
        return int(age + (row['TIMESTAMP'] - timestamp) / np.timedelta64(1, 'Y'))
    else:
        return np.nan

