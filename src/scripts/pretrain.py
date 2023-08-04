import hydra
import src.common.loading as loading
import src.common.create as create
from src.model.model import BertEHRModel
from src.trainer.trainer import EHRTrainer


@hydra.main(config_path="../../configs/train", config_name="pretrain")
def main_train(cfg):
    # Load in checkpoint if provided (cont. training)
    if cfg.paths.checkpoint is not None:
        cfg, checkpoint = loading.checkpoint(cfg)

    # Create datasets
    train_dataset, val_dataset = create.mlm_dataset(cfg)

    # Initialize model and optimizer
    model, optimizer = create.model(
        BertEHRModel, cfg, vocabulary=train_dataset.vocabulary
    )

    # Override state_dicts if checkpoint
    if cfg.paths.checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Initialize Trainer
    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main_train()
