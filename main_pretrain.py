from os.path import join

import torch
from torch.optim import AdamW
from transformers import BertConfig

from common.config import load_config
from model.model import BertEHRModel
from trainer.trainer import EHRTrainer

import logging


config_path = join("configs", "pretrain.yaml")
cfg = load_config(config_path)

def main_train(cfg):

    logging.basicConfig(filename=join(cfg.output_dir, 'info.log'), level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info('Loading data')
    train_dataset = torch.load(cfg.paths.train_dataset)
    val_dataset = torch.load(cfg.paths.val_dataset)
    vocabulary = torch.load(cfg.paths.vocabulary)
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
    logger.info('Initialize trainer')
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger
    )
    logger.info('Start training')
    trainer.train()


if __name__ == '__main__':
    main_train(cfg)