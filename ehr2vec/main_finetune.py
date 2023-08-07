import os
from os.path import join, 

import pandas as pd
from common.config import load_config
from common import azure
from common.setup import setup_run_folder
from common.loader import create_binary_outcome_datasets, load_model

from model.model import BertForFineTuning
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from trainer.trainer import EHRTrainer

config_path = join("configs", "finetune_test.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

run_name = "finetune"

def get_sampler_and_pos_weight(train_dataset, outcomes,cfg):
    sampler = None
    pos_weight = None
    if cfg.trainer_args['sampler']:
        labels = pd.Series(outcomes).notna().astype(int)
        label_weight = 1 / labels.value_counts()
        weights = labels.map(label_weight).values
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    elif cfg.trainer_args['pos_weight']:
        pos_weight = sum(pd.isna(outcomes)) / sum(pd.notna(outcomes))
    return sampler, pos_weight

def main_finetune():
    cfg = load_config(config_path)
    run = None

    if cfg.env=='azure':
        run, mount_context = azure.setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
    # Finetune specific
    logger = setup_run_folder(cfg)
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info(f'Outcome file: {cfg.paths.outcome}, Outcome name: {cfg.outcome.type}')
    logger.info(f'Censor file: {cfg.paths.censor}, Censor name: {cfg.outcome.censor_type}')
    logger.info(f"Censoring {cfg.outcome.n_hours} hours after censor_outcome")
    train_dataset, val_dataset, outcomes = create_binary_outcome_datasets(cfg)
    
    sampler, pos_weight = get_sampler_and_pos_weight(train_dataset, outcomes, cfg)

    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, pos_weight=pos_weight)
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
        sampler=sampler,
        cfg=cfg,
        run=run,
    )
    trainer.train()


if __name__ == '__main__':
    main_finetune()