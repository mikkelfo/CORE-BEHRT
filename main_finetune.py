import torch
import hydra
from torch.optim import AdamW
from data.dataset import CensorDataset
from trainer.trainer import EHRTrainer
from model.model import BertForFineTuning
from transformers import BertConfig

@hydra.main(version_base=None, config_path="configs/train", config_name="finetune")
def main_finetune(cfg):
    # Finetune specific
    train_encoded = torch.load(cfg.path.train_encoded)
    train_outcomes = torch.load(cfg.path.train_outcomes)
    val_encoded = torch.load(cfg.path.val_encoded)
    val_outcomes = torch.load(cfg.path.val_outcomes)
    n_hours, outcome_type, censor_type = cfg.outcome.n_hours, cfg.outcome.censor_type, cfg.outcome.type
    train_dataset = CensorDataset(train_encoded, n_hours=n_hours, outcomes=train_outcomes[outcome_type], censor_outcomes=train_outcomes[censor_type])
    val_dataset = CensorDataset(val_encoded, n_hours=n_hours, outcomes=val_outcomes[outcome_type], censor_outcomes=val_outcomes[censor_type])

    print(f'Setting up finetune task on [{outcome_type}] with [{n_hours}] hours censoring')

    model = BertForFineTuning(
        BertConfig(
            **cfg.model,
        )
    )
    model.load_state_dict(torch.load(cfg.path.pretrained_model))

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
    )
    trainer.train()


if __name__ == '__main__':
    main_finetune()