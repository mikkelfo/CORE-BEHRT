"""Pretrain hierarchical model on EHR data. Use config_template h_pretrain.yaml. Run setup_hierarchical.py first to create the vocabulary and tree."""
import os
from os.path import join

import torch
from ehr2vec.common.azure import AzurePathContext, save_to_blobstore
from ehr2vec.common.config import load_config
from ehr2vec.common.initialize import Initializer
from ehr2vec.common.loader import (load_checkpoint_and_epoch,
                           load_model_cfg_from_checkpoint)
from ehr2vec.common.setup import DirectoryPreparer, copy_data_config, get_args
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.model.config import adjust_cfg_for_behrt
from ehr2vec.trainer.trainer import EHRTrainer

CONFIG_NAME = 'h_pretrain.yaml'
BLOBSTORE = 'PHAIR'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_train(config_path):
    cfg = load_config(config_path)

    cfg, run, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).adjust_paths_for_azure_pretrain()
    
    logger, run_folder = DirectoryPreparer.setup_run_folder(cfg)
    copy_data_config(cfg, run_folder)
    
    load_model_cfg_from_checkpoint(cfg, 'pretrain_config.yaml') # if we are training from checkpoint, we need to load the old config
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_hmlm_dataset()

    torch.save(train_dataset.target_mapping, join(cfg.paths.output_path, cfg.paths.run_name, 'target_mapping.pt'))
    logger.info("Setup model")
    if cfg.model.get('behrt_embeddings', False):
        if cfg.paths.get('model_path', None) is None: # only if we are not training from checkpoint
            cfg = adjust_cfg_for_behrt(cfg)
    if cfg.model.get('medbert', False):
        if cfg.paths.get('model_path', None) is None:
            cfg.model.embedding = 'medbert'
    checkpoint, epoch = load_checkpoint_and_epoch(cfg)

    initializer = Initializer(cfg, checkpoint=checkpoint)
    model = initializer.initialize_hierachical_pretrain_model(train_dataset)
    logger.info("Setup optimizer and scheduler")
    optimizer = initializer.initialize_optimizer(model)
    scheduler = initializer.initialize_scheduler(optimizer)
        
    logger.info("Setup trainer")
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
        last_epoch=epoch
    )
    logger.info("Start training")
    trainer.train()
    if cfg.env == 'azure':
        save_to_blobstore(cfg.paths.run_name,
                          join(BLOBSTORE, 'models', cfg.paths.type, cfg.paths.run_name))
        mount_context.stop()
    logger.info("Done")


if __name__ == '__main__':
    main_train(config_path)