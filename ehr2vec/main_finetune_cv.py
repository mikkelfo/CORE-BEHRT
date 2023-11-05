import os
from os.path import abspath, dirname, join
from typing import List

import pandas as pd
import torch
from common.azure import save_to_blobstore
from common.initialize import Initializer, ModelManager
from common.setup import DirectoryPreparer, copy_data_config, get_args
from common.utils import Data
from data.dataset import BinaryOutcomeDataset
from data.prepare_data import DatasetPreparer
from data.split import get_n_splits_cv
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 2  # You can change this to desired value
BLOBSTORE='PHAIR'

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def finetune_fold(cfg, data:Data, train_indices:List[int], val_indices:List[int], fold:int)->None:
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
    dataset_preparer.saver.save_patient_nums(train_data, val_data, folder=fold_folder)

    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)
    
    modelmanager = ModelManager(cfg, fold)
    checkpoint = modelmanager.load_checkpoint() 
    modelmanager.load_model_config() # check whether cfg is changed after that.
    model = modelmanager.initialize_finetune_model(checkpoint, train_dataset)
    
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(model, train_dataset)
    epoch = modelmanager.get_epoch()


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
        last_epoch=epoch
    )
    trainer.train()
    

def compute_validation_scores_mean_std(finetune_folder: str)->None:
    """Compute mean and std of validation scores."""
    logger.info("Compute mean and std of validation scores")
    val_scores = []
    for fold in range(1, N_SPLITS+1):
        fold_checkpoints_folder = join(finetune_folder, f'fold_{fold}', 'checkpoints')
        last_epoch = max([int(f.split('_')[-1][:-4]) for f in os.listdir(fold_checkpoints_folder) if f.startswith('validation_scores')])
        fold_validation_scores = pd.read_csv(join(fold_checkpoints_folder, f'validation_scores_{last_epoch}.csv'))
        val_scores.append(fold_validation_scores)
    val_scores = pd.concat(val_scores)
    val_scores_mean_std = val_scores.groupby('metric')['value'].agg(['mean', 'std'])
    val_scores_mean_std.to_csv(join(finetune_folder, 'validation_scores_mean_std.csv'))

if __name__ == '__main__':

    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    
    for fold, (train_indices, val_indices) in enumerate(get_n_splits_cv(data, N_SPLITS)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        finetune_fold(cfg, data, train_indices, val_indices, fold)
        
    compute_validation_scores_mean_std(finetune_folder)
    
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
