import os
from datetime import datetime
from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.utils import resample
from torch.utils.data import WeightedRandomSampler

from ehr2vec.common.config import get_function
from ehr2vec.common.utils import Data

    
def validate_outcomes(all_outcomes, cfg):
    for outcome in cfg.outcomes:
        cfg.outcome = cfg.outcomes[outcome]
        if cfg.outcome.type:
            assert cfg.outcome.type in all_outcomes, f"Outcome {cfg.outcome.type} not found in outcomes."
        if cfg.outcome.get('censor_type', False):
            assert cfg.outcome.censor_type in all_outcomes, f"Censor type {cfg.outcome.censor_type} not found in outcomes."

def get_sampler(cfg, train_dataset, outcomes):
    """Get sampler for training data.
    sample_weight: float. Adjusts the number of samples in the positive class.
    """
    if cfg.trainer_args['sampler']:
        labels = pd.Series(outcomes).notna().astype(int)
        value_counts = labels.value_counts()
        label_weight = get_function(cfg.trainer_args['sample_weight_function'])(value_counts)
        label_weight[1] *= cfg.trainer_args.get('sample_weight_multiplier', 1.0) 
        weights = labels.map(label_weight).values
        # Adjust the weight for the positive class (1) using the sample_weight
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        return sampler
    else:
        return None
def inverse(x):
    return 1/x
def inverse_sqrt(x):
    return 1/np.sqrt(x)

def get_pos_weight(cfg, outcomes):
    if cfg.trainer_args.get('pos_weight', False):
        return sum(pd.isna(outcomes)) / sum(pd.notna(outcomes))
    else:
        return None

def compute_and_save_scores_mean_std(n_splits:int, finetune_folder: str, mode='val')->None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    scores = []
    for fold in range(1, n_splits+1):
        fold_checkpoints_folder = join(finetune_folder, f'fold_{fold}', 'checkpoints')
        last_epoch = max([int(f.split("_")[-2].split("epoch")[-1]) for f in os.listdir(fold_checkpoints_folder) if f.startswith('checkpoint_epoch')])
        table_path = join(fold_checkpoints_folder, f'{mode}_scores_{last_epoch}.csv')
        if not os.path.exists(table_path):
            continue
        fold_scores = pd.read_csv(join(fold_checkpoints_folder, f'{mode}_scores_{last_epoch}.csv'))
        scores.append(fold_scores)
    scores = pd.concat(scores)
    scores_mean_std = scores.groupby('metric')['value'].agg(['mean', 'std'])
    date = datetime.now().strftime("%Y%m%d-%H%M")
    scores_mean_std.to_csv(join(finetune_folder, f'{mode}_scores_mean_std_{date}.csv'))

def check_data_for_overlap(train_data: Data, val_data: Data, test_data: Data=None)->None:
    """Check that there is no overlap between train, val and test data"""
    train_pids = set(train_data.pids)
    val_pids = set(val_data.pids)
    assert len(train_pids.intersection(val_pids)) == 0, "Train and val data overlap"
    if test_data is not None:
        test_pids = set(test_data.pids) if len(test_data) > 0 else set()
        assert len(train_pids.intersection(test_pids)) == 0, "Train and test data overlap"
        assert len(val_pids.intersection(test_pids)) == 0, "Val and test data overlap"

def check_predefined_pids(data :Data, cfg)->None:
    if 'predefined_splits' in cfg.paths:
        all_predefined_pids = torch.load(join(cfg.paths.predefined_splits, 'pids.pt'))
        if not set(all_predefined_pids).issubset(set(data.pids)):
            difference = len(set(all_predefined_pids).difference(set(data.pids)))
            raise ValueError(f"Pids in the predefined splits must be a subset of data.pids. There are {difference} pids in the data that are not in the predefined splits")

def split_test_set(indices:list, test_split:float)->Tuple[list, list]:
    """Split intro test and train_val indices"""
    np.random.seed(42)
    test_indices = np.random.choice(indices, size=int(len(indices)*test_split), replace=False)
    test_indices_set = set(test_indices)
    train_val_indices = [i for i in indices if i not in test_indices_set]
    return test_indices, train_val_indices

def split_into_test_data_and_train_val_indices(cfg, data:Data)->Tuple[Data, list]:
    """Split data into test and train_val indices. And save test set."""
    indices = list(range(len(data.pids)))
    test_split = cfg.data.get('test_split', None)
    if test_split is not None:
        test_indices, train_val_indices = split_test_set(indices, test_split)
    else:
        test_indices = []
        train_val_indices = indices
    test_data = Data() if len(test_indices) == 0 else data.select_data_subset_by_indices(test_indices, mode='test')
    return test_data, train_val_indices

def save_data(data: Data, folder:str)->None:
    """Unpacks data and saves it to folder"""
    if len(data)>0:
        torch.save(data.pids, join(folder, f'{data.mode}_pids.pt'))
        torch.save(data.features, join(folder, f'{data.mode}_features.pt'))
        if data.outcomes is not None:
            torch.save(data.outcomes, join(folder, f'{data.mode}_outcomes.pt'))
        if data.censor_outcomes is not None:
            torch.save(data.censor_outcomes, join(folder, f'{data.mode}_censor_outcomes.pt'))
    else:
        raise Warning(f"Data is empty for {data}")