"""Pretrain hierarchical model on EHR data. Use config_template h_pretrain.yaml. Run setup_hierarchical.py first to create the vocabulary and tree."""
import os
from os.path import join

import torch
from common import azure
from common.config import load_config
from common.loader import create_hierarchical_dataset
from common.setup import setup_run_folder
from model.model import HierarchicalBertForPretraining
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig

config_path = 'configs/h_pretrain.yaml'
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
run_name = "h_pretrain"

def load_hierarchical_data(cfg):
    """Load hierarchical data from disk"""
    vocab = torch.load(join(cfg.paths.data_path, 'vocabulary.pt'))
    tree = torch.load(join(cfg.paths.data_path, 'hierarchical', 'tree.pt'))
    train_dataset, val_dataset = create_hierarchical_dataset(cfg)

    return vocab, tree, train_dataset, val_dataset


def main_train(config_path):
    cfg = load_config(config_path)
    
    if cfg.env=='azure':
        run, mount_context = azure.setup_azure(run_name)
        cfg.paths.output_path = join(mount_context.mount_point, cfg.paths.output_path)
    else:
        run = None
        cfg.paths.output_path = join('outputs', cfg.paths.output_path)
    
    logger = setup_run_folder(cfg)
    
    logger.info(f'Loading data from {cfg.paths.data_path}')
    vocab, tree, train_dataset, val_dataset = load_hierarchical_data(cfg)
  
    logger.info("Setup model")
    bertconfig = BertConfig(leaf_size=len(train_dataset.leaf_counts), vocab_size=len(vocab), **cfg.model)
    model = HierarchicalBertForPretraining(bertconfig, tree=tree)

    logger.info("Setup optimizer")
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )

    logger.info("Setup trainer")
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        run=run
    )
    logger.info("Start training")
    trainer.train()
    if cfg.env == 'azure':
        mount_context.stop()

if __name__ == '__main__':
    main_train(config_path)