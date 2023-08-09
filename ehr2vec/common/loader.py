import glob
import os
from os.path import join
import torch
from data.dataset import HierarchicalMLMDataset, MLMDataset, CensorDataset
from transformers import BertConfig

def load_model(model_class, cfg, add_config={}):
    checkpoint_path = join(cfg.paths.model_path, 'checkpoints' ,f'checkpoint_epoch{cfg.paths.checkpoint_epoch}_end.pt')
    # Load the config from file
    config = BertConfig.from_pretrained(cfg.paths.model_path) 
    config.update(add_config)
    model = model_class(config)
    load_result = model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)
    print("missing state dict keys", load_result.missing_keys)
    return model

def create_binary_outcome_datasets(cfg):
    """
    This function is used to create outcome datasets based on the configuration provided.
    """
    outcomes, censor_outcomes, pids = load_outcomes(cfg)
    train_dataset = CensorDataset(cfg.paths.data_path, 'train', outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.train_data.num_patients,
                                    n_hours=cfg.outcome.n_hours,)
    val_dataset = CensorDataset(cfg.paths.data_path, 'val',  outcomes, 
                                    censor_outcomes=censor_outcomes, 
                                    outcome_pids=pids,
                                    num_patients=cfg.val_data.num_patients,
                                    n_hours=cfg.outcome.n_hours,
                                    )
    
    return train_dataset, val_dataset, outcomes

def get_val_test_pids(cfg):
    """Gets the pretrain validation pids and splits into train and test pids for finetuning."""
    # !Currently unused
    val_pids = torch.load(join(cfg.paths.data_path, 'val_pids.pt'))
    val_pids = val_pids[:cfg.val_data.num_patients]
    test_cutoff = int(len(val_pids)*cfg.test_data.split)
    test_pids = val_pids[:test_cutoff]
    val_pids = val_pids[test_cutoff:]
    return val_pids, test_pids

def load_outcomes(cfg):
    """From the configuration, load the outcomes and censor outcomes.
    Access pids, the outcome of interest and the censoring outcome."""
    all_outcomes = torch.load(cfg.paths.outcomes_path)
    outcomes = all_outcomes.get(cfg.outcome.type, None)
    censor_outcomes = all_outcomes.get(cfg.outcome.censor_type, None)
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

def check_directory_for_features(dir_, logger):
    features_dir = join(dir_, 'features')
    if os.path.exists(features_dir):
        if len(glob.glob(join(features_dir, 'features_*.pt')))>0:
            logger.warning(f"Features already exist in {features_dir}.")
            logger.warning(f"Skipping feature creation.")
        return True
    else:
        return False