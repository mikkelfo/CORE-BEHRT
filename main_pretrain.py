from os.path import join

from torch.optim import AdamW
from transformers import BertConfig
from transformers import get_linear_schedule_with_warmup 

from common.config import load_config
from common.loader import create_datasets
from common.setup import setup_run_folder
from model.model import BertEHRModel
from trainer.trainer import EHRTrainer

# from azure_run.run import Run
# from azure_run import datastore

# from azureml.core import Dataset

config_path = join("configs", "pretrain.yaml")

# run = Run
run = None
# run.name(f"Pretrain base diagnosis medication")

# ds = datastore("workspaceblobstore")
# dataset = Dataset.File.from_files(path=(ds, 'PHAIR'))


def main_train(config_path):
    cfg = load_config(config_path)

    # mount dataset
    # mount_context = dataset.mount()
    # mount_context.start()  # this will mount the file streams
    
    # cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
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
    # mount_context.stop()


if __name__ == '__main__':
    main_train(config_path)