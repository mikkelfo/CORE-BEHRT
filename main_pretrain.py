"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""
import os
from os.path import join

from common import azure
from common.config import load_config
from common.loader import create_datasets
from common.setup import setup_run_folder
from model.model import BertEHRModel
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig, get_linear_schedule_with_warmup

config_path = join("configs", "pretrain.yaml")
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

run_name = "pretrain"

def main_train(config_path):
    cfg = load_config(config_path)
    if cfg.env=='azure':
        run, mount_context = azure.setup_azure(cfg.paths.run_name)
        cfg.paths.output_path = join(mount_context.mount_point, cfg.paths.output_path)
    else:
        run = None
        cfg.paths.output_path = join('outputs', cfg.paths.output_path)
    

    logger = setup_run_folder(cfg)
    logger.info(f"Access data from {cfg.paths.data_path}")
    logger.info('Loading data')
    
    train_dataset, val_dataset, vocabulary = create_datasets(cfg)
    
    logger.info('Initializing model')
    model = BertEHRModel(
        BertConfig(
            **cfg.model,
            vocab_size=len(vocabulary),

        )
    )
    logger.info('Initializing optimizer')
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.epsilon,
    )
    if cfg.scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.scheduler.num_warmup_steps, num_training_steps=cfg.scheduler.num_training_steps)
    logger.info('Initialize trainer')
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        run=run,
    )
    logger.info('Start training')
    trainer.train()
    if cfg.env == 'azure':
        setup['mount_context'].stop()

if __name__ == '__main__':
    main_train(config_path)