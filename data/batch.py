from os.path import join

import numpy as np
import torch
from tqdm import tqdm


def split_batches(num_batches, split_ratios):
    batches = np.arange(num_batches)
    np.random.shuffle(batches)
    # calculate the number of batches for each set
    train_end = int(split_ratios['train'] * len(batches))
    val_end = train_end + int(split_ratios['val'] * len(batches))
    # split the batches into train, validation and test
    train_batches = batches[:train_end]
    val_batches = batches[train_end:val_end]
    test_batches = batches[val_end:]
    return train_batches, val_batches, test_batches

def batch_tokenize(cfg, tokenizer, batches, mode='train'):
    files = []
    for batch in tqdm(batches, desc=f'Tokenizing {mode} batches'):
        features = torch.load(join(cfg.output_dir, f'features{batch}.pt'))
        train_encoded = tokenizer(features)
        torch.save(train_encoded, join(cfg.output_dir, f'encoded_{mode}{batch}.pt'))
        files.append(join(cfg.output_dir, f'encoded_{mode}{batch}.pt'))
    return files