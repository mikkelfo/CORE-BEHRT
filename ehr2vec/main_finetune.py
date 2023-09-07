import os
from os.path import join, split

import pandas as pd
import torch
from common.azure import setup_azure
from common.config import load_config
from common.loader import load_model, retrieve_outcomes
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
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f'Outcome file: {cfg.paths.outcome}, Outcome name: {cfg.outcome.type}')
    logger.info(f'Censor file: {cfg.paths.censor}, Censor name: {cfg.outcome.censor_type}')
    logger.info(f"Censoring {cfg.outcome.n_hours} hours after censor_outcome")
    all_outcomes = torch.load(join(cfg.paths.data_path, cfg.paths.outcome))

    outcomes, censor_outcomes, pids = retrieve_outcomes(all_outcomes, cfg)
    # TODO: that should be configurable
    pids = filter_pids(cfg.paths.model_path, pids, logger)
    train_dataset, val_dataset = get_datasets(cfg, outcomes, censor_outcomes, pids)

    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, {'pos_weight':get_pos_weight(cfg, outcomes)})
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
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

def get_datasets(cfg, outcomes, censor_outcomes, pids):
    kwargs = {
        'data_dir': cfg.paths.data_path,
        'outcomes': outcomes,
        'censor_outcomes': censor_outcomes,
        'outcome_pids': pids,
        'pids': pids,
        'n_hours': cfg.outcome.n_hours,
        'n_procs': cfg.train_data.get('n_procs', None),
        'truncation_len': cfg.get('truncation_len', None),
    }
    train_dataset = CensorDataset(mode='train', num_patients=cfg.train_data.num_patients, **kwargs)
    val_dataset = CensorDataset(mode='val', num_patients=cfg.val_data.num_patients, **kwargs)
    return train_dataset, val_dataset

if __name__ == '__main__':
    main_finetune()