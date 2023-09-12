import os
from os.path import join, split

import pandas as pd
import torch
from common.azure import setup_azure
from common.config import load_config
from common.loader import (load_model, load_tokenized_data, retrieve_outcomes,
                           select_patient_subset)
from common.setup import get_args, setup_run_folder
from data.dataset import CensorDataset
from model.model import BertForFineTuning
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from trainer.trainer import EHRTrainer

args = get_args('finetune.yaml')

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_finetune():
    cfg = load_config(config_path)
    run = None

    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
        cfg.paths.output_path = join("outputs")
    # Finetune specific
    logger = setup_run_folder(cfg)
    logger.info(f"Access data from {cfg.paths.data_path}, Outcome file: {cfg.paths.outcome}, Outcome name: {cfg.outcome.type}")
    logger.info(f'Censor file: {cfg.paths.censor}, Censoring {cfg.outcome.n_hours} hours after {cfg.outcome.censor_type}')
     
    train_features, train_pids, val_features, val_pids, vocabulary = load_tokenized_data(cfg)
    train_features, train_pids = exclude_pretrain_patients(train_features, train_pids, cfg, 'train')
    val_features, val_pids = exclude_pretrain_patients(val_features, val_pids, cfg, 'val')
    train_features, train_pids, val_features, val_pids = select_patient_subset(train_features, train_pids, val_features, val_pids, cfg.train_data.num_patients, cfg.val_data.num_patients)
    
    outcomes = torch.load(join(cfg.paths.data_path, cfg.paths.outcome))
    outcomes_train = select_outcomes_for_patients(outcomes, train_pids)
    outcomes_val = select_outcomes_for_patients(outcomes, val_pids)
    outcome_train, censor_outcome_train, _ = retrieve_outcomes(outcomes_train, cfg)
    outcome_val, censor_outcome_val, _ = retrieve_outcomes(outcomes_val, cfg)

    train_dataset = CensorDataset(
            features=train_features, outcomes=outcome_train, 
            censor_outcomes=censor_outcome_train, vocabulary=vocabulary, 
            n_hours=cfg.outcome.n_hours,
            truncation_len=cfg.dataset.get('truncation_len', None))
    val_dataset = CensorDataset(
        features=val_features, outcomes=outcome_val, 
        censor_outcomes=censor_outcome_val, vocabulary=vocabulary, 
        n_hours=cfg.outcome.n_hours,
        truncation_len=cfg.dataset.get('truncation_len', None))
   
    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, {'pos_weight':get_pos_weight(cfg, outcomes)})
    optimizer = AdamW(
        model.parameters(),
        **cfg.optimizer
    )

    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=get_sampler(cfg, train_dataset, outcomes, logger),
        cfg=cfg,
        run=run,
    )
    trainer.train()
    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", 'models', split(cfg.paths.model_path)[-1], "finetune_"+cfg.run_name))
        mount_context.stop()
    logger.info('Done')


def get_sampler(cfg, train_dataset, outcomes, logger):
    if cfg.trainer_args['sampler']:
        logger.warning('Sampler does not work with IterableDataset. Use positive weight instead')
        labels = pd.Series(outcomes).notna().astype(int)
        label_weight = 1 / labels.value_counts()
        weights = labels.map(label_weight).values
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        return sampler
    else:
        return None

def get_pos_weight(cfg, outcomes):
    if cfg.trainer_args['pos_weight']:
        return sum(pd.isna(outcomes)) / sum(pd.notna(outcomes))
    else:
        return None

def filter_pids(model_path, pids, logger):
    """Filter pids based on the pids that have already been trained on."""
    logger.info('Filtering pids')
    train_pids = torch.load(join(model_path, 'train_pids.pt'))
    val_pids = torch.load(join(model_path, 'val_pids.pt'))
    pt_pids = set(train_pids).union(set(val_pids)) 
    logger.info(f"Number of pids before filtering: {len(pids)}")
    ft_pids = [pid for pid in pids if pid not in pt_pids]
    return ft_pids

def select_outcomes_for_patients(all_outcomes, pids):
    pids = set(pids)
    return {k:[v[i] for i, pid in enumerate(all_outcomes['PID']) if pid in pids] for k, v in all_outcomes.items()}

def exclude_pretrain_patients(features, pids, cfg, mode):
    pretrain_pids = set(torch.load(join(cfg.paths.model_path, f'pids_{mode}.pt')))
    kept_indices = [i for i, pid in enumerate(pids) if pid not in pretrain_pids]
    pids = [pid for i, pid in enumerate(pids) if i in kept_indices]
    for k, v in features.items():
        features[k] = [v[i] for i in kept_indices]
    return features, pids

if __name__ == '__main__':
    main_finetune()