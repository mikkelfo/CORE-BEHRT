import hydra
import src.common.loading as loading
import src.common.create as create
from src.model.model import BertForFineTuning
from src.trainer.trainer import EHRTrainer
from omegaconf import open_dict


@hydra.main(config_path="../../configs/train", config_name="finetune")
def main_finetune(cfg):
    # Create datasets
    train_dataset, val_dataset = create.censor_dataset(cfg)

    # Setup sampler
    sampler, pos_weight = create.sampler(cfg, train_dataset)

    # Give info about finetuning process
    print(
        f'Setting up finetune task on [{cfg.outcome.type}] with [{cfg.outcome.n_hours}] hours censoring at [{cfg.outcome.censor_type}] \
            using pos_weight [{pos_weight}] and sampler [{cfg.trainer_args["sampler"]}]'
    )

    # Get pretrained checkpoint and overwrite cfg.model
    cfg_cp, checkpoint = loading.checkpoint(cfg)
    with open_dict(cfg):  # Opens config to add new key
        cfg.model = cfg_cp.model

    # Initialize model and optimizer
    model, optimizer = create.model(BertForFineTuning, cfg, pos_weight=pos_weight)

    # Load in pretrained model
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Initialize Trainer
    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        cfg=cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main_finetune()
