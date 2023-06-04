from os.path import join

from torch.optim import AdamW
from transformers import BertConfig

from common.config import load_config
from common.loader import create_datasets
from common.setup import setup_run_folder
from model.model import BertEHRModel
from trainer.trainer import EHRTrainer

config_path = join("configs", "pretrain.yaml")
cfg = load_config(config_path)



def main_train(cfg):

    logger = setup_run_folder(cfg)
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