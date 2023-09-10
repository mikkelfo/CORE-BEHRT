"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""
import os
from os.path import join
import torch

from common.azure import setup_azure
from common.config import load_config
from common.loader import load_tokenized_data, select_patient_subset, save_pids
from common.setup import setup_run_folder, get_args
from model.model import BertEHRModel
from torch.optim import AdamW
from data.dataset import MLMDataset
from trainer.trainer import EHRTrainer
from transformers import BertConfig, get_linear_schedule_with_warmup


args = get_args('pretrain.yaml')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_train(config_path):
    cfg = load_config(config_path)
    run = None
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)

    logger = setup_run_folder(cfg)
    
    
    logger.info(f"Loading data from {cfg.paths.data_path} with {cfg.train_data.get('num_patients', 'all')} train patients and {cfg.val_data.get('num_patients', 'all')} val patients")
    train_features, train_pids, val_features, val_pids, vocabulary = load_tokenized_data(cfg)
    train_features, train_pids, val_features, val_pids = select_patient_subset(train_features, train_pids, val_features, val_pids, cfg.train_data.num_patients, cfg.val_data.num_patients)
    run_folder = join(cfg.paths.output_path, cfg.paths.run_name)
    save_pids(run_folder, train_pids, val_pids)
    torch.save(vocabulary, join(run_folder, 'vocabulary.pt'))
    train_dataset = MLMDataset(train_features, vocabulary, **cfg.dataset)
    val_dataset = MLMDataset(val_features, vocabulary, **cfg.dataset)


    logger.info('Initializing model')
    model = BertEHRModel(
        BertConfig(
            **cfg.model,
            vocab_size=len(train_dataset.vocabulary),

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
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", 'models', cfg.paths.type, cfg.paths.run_name))
        mount_context.stop()
    logger.info("Done")
if __name__ == '__main__':
    main_train(config_path)