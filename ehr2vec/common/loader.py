import glob
import logging
import os
from os.path import join

import numpy as np
import pandas as pd
import torch
from data.dataset import CensorDataset, HierarchicalMLMDataset, MLMDataset
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module

def load_model(model_class, cfg, add_config={}):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = join(cfg.paths.model_path, 'checkpoints' ,f'checkpoint_epoch{cfg.paths.checkpoint_epoch}_end.pt')
    # Load the config from file
    config = BertConfig.from_pretrained(cfg.paths.model_path) 
    config.update(add_config)
    model = model_class(config)
    load_result = model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'], strict=False)
    print("missing state dict keys", load_result.missing_keys)
    return model

def create_binary_outcome_datasets(all_outcomes, cfg):
    """
    This function is used to create outcome datasets based on the configuration provided.
    """
    outcomes, censor_outcomes, pids = retrieve_outcomes(all_outcomes, cfg)
    if cfg.get("encode_pos_only", False):
        outcomes, censor_outcomes, pids = select_positives(outcomes, censor_outcomes, pids)
        cfg.train_data.num_patients = None
        cfg.val_data.num_patients = None
    if cfg.train_data.num_patients == 0:
        train_dataset = None
    else:
        train_dataset = CensorDataset(cfg.paths.data_path, 'train', outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.train_data.num_patients,
                                    pids=pids if cfg.get("encode_pos_only", False) else None,
                                    n_hours=cfg.outcome.n_hours,
                                    n_procs=cfg.train_data.get('n_procs', None))

    if cfg.val_data.num_patients == 0:
        val_dataset = None
    else:
        val_dataset = CensorDataset(cfg.paths.data_path, 'val',  outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.val_data.num_patients,
                                    pids=pids if cfg.get("encode_pos_only", False) else None, 
                                    n_hours=cfg.outcome.n_hours,
                                    n_procs=cfg.val_data.get('n_procs', None),
                                    )
    
    return train_dataset, val_dataset, outcomes

def select_positives(outcomes, censor_outcomes, pids):
    """Select only positive outcomes."""
    logger.info("Selecting only positive outcomes")
    select_indices = [i for i, outcome in enumerate(outcomes) if pd.notna(outcome)]
    outcomes = [outcomes[i] for i in select_indices]
    censor_outcomes = [censor_outcomes[i] for i in select_indices]
    pids = [pids[i] for i in select_indices]
    return outcomes, censor_outcomes, pids

def get_val_test_pids(cfg):
    """Gets the pretrain validation pids and splits into train and test pids for finetuning."""
    # !Currently unused
    val_pids = torch.load(join(cfg.paths.data_path, 'val_pids.pt'))
    val_pids = val_pids[:cfg.val_data.num_patients]
    test_cutoff = int(len(val_pids)*cfg.test_data.split)
    test_pids = val_pids[:test_cutoff]
    val_pids = val_pids[test_cutoff:]
    return val_pids, test_pids

def retrieve_outcomes(all_outcomes, cfg):
    """From the configuration, load the outcomes and censor outcomes.
    Access pids, the outcome of interest and the censoring outcome."""
    
    outcomes = all_outcomes.get(cfg.outcome.type, [None]*len(all_outcomes['PID']))
    censor_outcomes = all_outcomes.get(cfg.outcome.get('censor_type', None), [None]*len(outcomes))
    pids = all_outcomes['PID']
    return outcomes, censor_outcomes, pids

def create_datasets(cfg, hierarchical:bool=False):
    """
    This function is used to create datasets based on the configuration provided.
    """
    if hierarchical:
        DatasetClass = HierarchicalMLMDataset
    else:
        DatasetClass = MLMDataset
    train_dataset, val_dataset = load_datasets(cfg, DatasetClass)
    return train_dataset, val_dataset

def load_datasets(cfg, DS):
    """
    This function is used to load datasets based on the given DatasetClass and configuration.
    """
    data_path = cfg.paths.data_path
    train_dataset = DS(data_path, 'train', **cfg.dataset, **cfg.train_data)
    val_dataset = DS(data_path, 'val', **cfg.dataset, **cfg.val_data)
    return train_dataset, val_dataset

def check_directory_for_features(dir_):
    features_dir = join(dir_, 'features')
    if os.path.exists(features_dir):
        if len(glob.glob(join(features_dir, 'features_*.pt')))>0:
            logger.warning(f"Features already exist in {features_dir}.")
            logger.warning(f"Skipping feature creation.")
        return True
    else:
        return False
    
# New setup with map-style datasets
def load_tokenized_data(cfg):
    tokenized_dir = cfg.paths.get('tokenized_dir', 'tokenized')
    tokenized_data_path = join(cfg.paths.data_path, tokenized_dir)
    logger.info("Loading tokenized data train")
    train_features  = torch.load(join(tokenized_data_path, 'tokenized_train.pt'))
    train_pids = torch.load(join(tokenized_data_path,  'pids_train.pt'))
    logger.info("Loading tokenized data val")
    val_features = torch.load(join(tokenized_data_path, 'tokenized_val.pt'))
    val_pids = torch.load(join(tokenized_data_path, 'pids_val.pt'))
    logger.info("Loading vocabulary")
    try:
        vocabulary = torch.load(join(tokenized_data_path, 'vocabulary.pt'))
    except:
        vocabulary = torch.load(join(cfg.paths.data_path, 'vocabulary.pt'))
    return train_features, train_pids, val_features, val_pids, vocabulary
    
def load_tree(cfg, hierarchical_dir='hierarchical'):
    hierarchical_path = join(cfg.paths.data_path, hierarchical_dir)
    tree = torch.load(join(hierarchical_path, 'tree.pt'))
    tree_matrix = torch.load(join(hierarchical_path, 'tree_matrix.pt'))
    h_vocabulary = torch.load(join(hierarchical_path, 'vocabulary.pt'))
    return tree, tree_matrix, h_vocabulary

def select_patient_subset(train_features, train_pids, val_features, val_pids, train_num_patients=None, val_num_patients=None):
    if train_num_patients is not None:
        train_features, train_pids = select_random_subset(train_features, train_pids, train_num_patients)
    if val_num_patients is not None:
        val_features, val_pids = select_random_subset(val_features, val_pids, val_num_patients)
    return train_features, train_pids, val_features, val_pids

def select_random_subset(features, pids, num_patients, seed=0):#
    np.random.seed(seed)
    indices = np.arange(len(pids))
    np.random.shuffle(indices)
    indices = indices[:num_patients]
    pids = [pid for i, pid in enumerate(pids) if i in indices]
    for k, v in features.items():
        features[k] = [v[i] for i in indices]
    return features, pids

def save_pids(folder, train_pids, val_pids):
    torch.save(train_pids, join(folder, 'pids_train.pt'))
    torch.save(val_pids, join(folder, 'pids_val.pt'))