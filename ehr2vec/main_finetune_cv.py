import os
from os.path import join, dirname, abspath
from shutil import copyfile
from typing import Iterator, List, Tuple

import torch
from common.config import instantiate, load_config
from common.loader import DatasetPreparer, load_model
from common.setup import (add_pretrain_info_to_cfg, adjust_paths_for_finetune,
                          azure_finetune_setup, get_args, setup_logger,
                          setup_run_folder)
from common.utils import Data
from data.dataset import BinaryOutcomeDataset
from evaluation.utils import get_pos_weight, get_sampler
from model.model import BertForFineTuning
from sklearn.model_selection import KFold
from torch.optim import AdamW
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 5  # You can change this to desired value

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def finetune_fold(data:Data, train_indices:List[int], val_indices:List[int], fold:int)->None:
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Splitting data")
    train_data = data.select_data_subset_by_indices(train_indices, mode='train')
    val_data = data.select_data_subset_by_indices(val_indices, mode='val')

    logger.info("Saving pids")
    torch.save(train_data.pids, join(fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(fold_folder, 'val_pids.pt'))

    logger.info("Saving patient numbers")
    dataset_preparer.save_patient_nums(train_data, val_data, folder=fold_folder)

    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)

    logger.info('Initializing model')
    model = load_model(BertForFineTuning, cfg, 
                       {'pos_weight':get_pos_weight(cfg, train_dataset.outcomes),
                        'embedding':'original_behrt' if cfg.model.get('behrt_embeddings', False) else None,
                        'pool_type': cfg.model.get('pool_type', 'mean')})

    try:
        logger.warning('Compilation currently leads to torchdynamo error during training. Skip it')
        #model = torch.compile(model)
        #logger.info('Model compiled')
    except:
        logger.info('Model not compiled')    

    optimizer = AdamW(model.parameters(), **cfg.optimizer)

    sampler = get_sampler(cfg, train_dataset, train_dataset.outcomes)
    if sampler:
        cfg.trainer_args.shuffle = False

    if cfg.scheduler:
        cfg.scheduler.optimizer = optimizer
        scheduler = instantiate(cfg.scheduler)

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
        run_folder=fold_folder,
    )
    trainer.train()
    
def get_n_splits(data: Data, n_splits: int)->Iterator[Tuple[Data,Data]]:
    """Get indices for n_splits cross validation."""
    kf = KFold(n_splits=n_splits)
    indices = list(range(len(data.pids)))
    for train_indices, val_indices in kf.split(indices):
        yield train_indices, val_indices

def initialize_configuration():
    """Load and adjust the configuration."""
    cfg = load_config(config_path)
    model_path = cfg.paths.model_path
    cfg = adjust_paths_for_finetune(cfg)
    cfg, run, mount_context = azure_finetune_setup(cfg)
    cfg = add_pretrain_info_to_cfg(cfg)
    return cfg, run, mount_context, model_path

def setup_logging_and_folders(cfg):
    """Setup the logger and relevant folders."""
    logger = setup_run_folder(cfg)

    run_name = cfg.paths.run_name 
    run_folder = join(cfg.paths.output_path, run_name)

    os.makedirs(run_folder, exist_ok=True)
    tokenized_dir_name = cfg.paths.get('tokenized_dir', 'tokenized')
    try:
        copyfile(join(cfg.paths.data_path, tokenized_dir_name, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))
    except:
        copyfile(join(cfg.paths.data_path, 'data_config.yaml'), join(run_folder, 'data_config.yaml'))

    logger = setup_logger(run_folder)
    logger.info(f'Run folder: {run_folder}')

    finetune_folder = join(cfg.paths.output_path, cfg.paths.run_name)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))

    return logger, finetune_folder

if __name__ == '__main__':

    cfg, run, mount_context, model_path = initialize_configuration()
    logger, finetune_folder = setup_logging_and_folders(cfg)

    logger.info("Prepare Features")
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    
    for fold, (train_indices, val_indices) in enumerate(get_n_splits(data, N_SPLITS)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        finetune_fold(data, train_indices, val_indices, fold)
        
    if cfg.env=='azure':
        logger.info("Saving results to blob")
        try:
            from azure_run import file_dataset_save
            file_dataset_save(local_path=join('outputs', cfg.paths.run_name), datastore_name = "workspaceblobstore",
                        remote_path = join("PHAIR", model_path, cfg.paths.run_name))
            logger.info("Saved to Azure Blob Storage")
        except Exception as e:
            logger.warning(f'Could not save to Azure Blob Storage. Error {e}')
        mount_context.stop()
    logger.info('Done')
