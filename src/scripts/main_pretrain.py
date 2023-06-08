import torch
import hydra
from transformers import BertConfig
from torch.optim import AdamW
from src.data.dataset import MLMDataset
from src.trainer.trainer import EHRTrainer
from src.model.model import BertEHRModel

@hydra.main(config_path="configs/train", config_name="pretrain")
def main_train(cfg):
    # MLM specific
    train_encoded = torch.load(cfg.paths.train_encoded)
    val_encoded = torch.load(cfg.paths.val_encoded)
    vocabulary = torch.load(cfg.paths.vocabulary)
    train_dataset = MLMDataset(train_encoded, vocabulary=vocabulary, ignore_special_tokens=cfg.ignore_special_tokens)
    val_dataset = MLMDataset(val_encoded, vocabulary=vocabulary, ignore_special_tokens=cfg.ignore_special_tokens)

    model = BertEHRModel(
        BertConfig(
            **cfg.model
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
    main_train()