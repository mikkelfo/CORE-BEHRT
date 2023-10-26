import os
from os.path import join

import torch
from common.azure import save_to_blobstore
from common.config import load_config
from common.initialize import Initializer
from common.loader import ModelLoader, load_model_cfg_from_checkpoint
from common.setup import (AzurePathContext, DirectoryPreparer,
                          copy_data_config, get_args)
from data.dataset import BinaryOutcomeDataset
from data.prepare_data import DatasetPreparer
from data.utils import Utilities
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
BLOBSTORE='PHAIR'
CHECKPOINTS_DIR = 'checkpoints'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_finetune():
    cfg = load_config(config_path)
    pretrain_model_path = cfg.paths.pretrain_model_path # ! do not change this, it is used for saving the model to blobstore
    
    cfg = DirectoryPreparer.adjust_paths_for_finetune(cfg) 
    azure_context = AzurePathContext(cfg)
    cfg, run, mount_context = azure_context.azure_finetune_setup()
    cfg = azure_context.add_pretrain_info_to_cfg()
    
    logger, run_folder = DirectoryPreparer.setup_run_folder(cfg)

    copy_data_config(cfg, run_folder)
    cfg.save_to_yaml(join(run_folder, 'finetune_config.yaml'))
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    
    train_data, val_data = data.split(cfg.data.get('val_split', 0.2))
    torch.save(train_data.pids, join(run_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(run_folder, 'val_pids.pt'))
    
    dataset_preparer.saver.save_patient_nums(train_data, val_data)
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)

    model_path = cfg.paths.get('model_path', None)
    checkpoint_model_path = model_path if model_path is not None else cfg.paths.pretrain_model_path    
    checkpoint = ModelLoader(cfg, checkpoint_model_path).load_checkpoint()
    if model_path:
        load_model_cfg_from_checkpoint(cfg, 'finetune_config.yaml') # if we are training from checkpoint, we need to load the old config
    initializer = Initializer(cfg, checkpoint=checkpoint, model_path=checkpoint_model_path)
    model = initializer.initialize_finetune_model(train_dataset)
    
    if model_path is None: # if no model_path provided, optimizer and scheduler are initialized from scratch
        initializer.checkpoint = None 
        epoch = 0
    else:
        epoch = Utilities.get_last_checkpoint_epoch(join(model_path, CHECKPOINTS_DIR)) 

    optimizer = initializer.initialize_optimizer(model)
    sampler, cfg = initializer.initialize_sampler(train_dataset)
    scheduler = initializer.initialize_scheduler(optimizer)

    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        scheduler=scheduler,
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        last_epoch=epoch
    )
    trainer.train()
    if cfg.env=='azure':
        save_to_blobstore(cfg.paths.run_name,
                          join(BLOBSTORE, pretrain_model_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')

if __name__ == '__main__':
    main_finetune()