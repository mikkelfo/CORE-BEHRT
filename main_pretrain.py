from os.path import join

import torch
from torch.optim import AdamW
from transformers import BertConfig

from data import utils
from model.model import BertEHRModel
from trainer.trainer import EHRTrainer


config_path = join("configs", "pretrain.yaml")
cfg = utils.load_config(config_path)

def main_train(cfg):
    # MLM specific
    train_dataset = torch.load(cfg.paths.train_dataset)
    val_dataset = torch.load(cfg.paths.val_dataset)
    vocabulary = torch.load(cfg.paths.vocabulary)

    model = BertEHRModel(
        BertConfig(
            **cfg.model,
            vocab_size=len(vocabulary),

        )
    )

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
        cfg=cfg
    )
    trainer.train()


if __name__ == '__main__':
    main_train(cfg)