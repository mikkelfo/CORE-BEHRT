"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""
import os
from os.path import join

from common.azure import save_to_blobstore, setup_azure
from common.config import load_config
from common.initialize import Initializer
from common.loader import DatasetPreparer, Loader, Utilities
from common.setup import copy_data_config, get_args, setup_run_folder
from model.config import adjust_cfg_for_behrt
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'pretrain.yaml'
BLOBSTORE = 'PHAIR'
CHECKPOINTS_FOLDER = 'checkpoints'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_train(config_path):
    cfg = load_config(config_path)
    
    run = None
    if cfg.env=='azure':
        run, mount_context = setup_azure(cfg.paths.run_name)
        adjust_paths_for_azure_pretrain(cfg)

    logger, run_folder = setup_run_folder(cfg)
    copy_data_config(cfg, run_folder)
    
    load_model_cfg_from_checkpoint(cfg) # if we are training from checkpoint, we need to load the old config
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_mlm_dataset()
    
    if cfg.model.get('behrt_embeddings', False):
        if cfg.paths.get('model_path', None) is None: # only if we are not training from checkpoint
            cfg = adjust_cfg_for_behrt(cfg)

    checkpoint, epoch = load_checkpoint_and_epoch(cfg)
    logger.info('Initializing model')
    initializer = Initializer(cfg, checkpoint=checkpoint)
    model = initializer.initialize_pretrain_model(vocab_size=len(train_dataset.vocabulary))

    logger.info('Initializing optimizer')
    optimizer = initializer.initialize_optimizer(model)
    scheduler = initializer.initialize_scheduler(optimizer)
        
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
        last_epoch=epoch
    )
    logger.info('Start training')
    trainer.train()
    if cfg.env == 'azure':
        save_to_blobstore(cfg.paths.run_name,
                          join(BLOBSTORE, 'models', cfg.paths.type, cfg.paths.run_name))
        mount_context.stop()
    logger.info("Done")

def adjust_paths_for_azure_pretrain(cfg, mount_context):
    """
    Adjusts the following paths in the configuration for the Azure environment:
    - data_path
    - model_path
    - output_path
    """
    cfg.paths.data_path = join(mount_context.mount_point, cfg.paths.data_path)
    if cfg.paths.get('model_path', None) is not None:
        cfg.paths.model_path = join(mount_context.mount_point, cfg.paths.model_path)
    if not cfg.paths.output_path.startswith('outputs'):
        cfg.paths.output_path = join('outputs', cfg.paths.output_path)
    
def load_checkpoint_and_epoch(cfg):
    model_path = cfg.paths.get('model_path', None)
    checkpoint = Loader(cfg).load_checkpoint() if model_path is not None else None
    epoch = Utilities.get_last_checkpoint_epoch(join(model_path, CHECKPOINTS_FOLDER)) if model_path is not None else None
    return checkpoint, epoch

def load_model_cfg_from_checkpoint(cfg):
    """If training from checkpoint, we need to get the old config"""
    model_path = cfg.paths.get('model_path', None)
    if model_path is not None: # if we are training from checkpoint, we need to load the old config
        old_cfg = load_config(join(cfg.paths.model_path, 'pretrain_config.yaml'))
        cfg.model = old_cfg.model


if __name__ == '__main__':
    main_train(config_path)
