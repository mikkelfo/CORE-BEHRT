import os
import random
from os.path import abspath, dirname, join
from typing import List

import torch
from common.azure import save_to_blobstore
from common.initialize import Initializer, ModelManager
from common.setup import (DirectoryPreparer, copy_data_config,
                          copy_pretrain_config, get_args)
from common.utils import Data
from data.dataset import BinaryOutcomeDataset
from data.prepare_data import DatasetPreparer
from data.split import get_n_splits_cv
from evaluation.utils import compute_and_save_scores_mean_std
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 2  # You can change this to desired value
BLOBSTORE='PHAIR'

random.seed(42)

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def finetune_fold(cfg, data:Data, train_indices:List[int], val_indices:List[int], 
                fold:int, test_indices: List[int]=[])->None:
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Splitting data")
    train_data = data.select_data_subset_by_indices(train_indices, mode='train')
    val_data = data.select_data_subset_by_indices(val_indices, mode='val')
    test_data = None if len(test_indices) == 0 else data.select_data_subset_by_indices(test_indices, mode='test')

    logger.info("Saving pids")
    torch.save(train_data.pids, join(fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(fold_folder, 'val_pids.pt'))
    if len(test_indices) > 0:
        test_data = data.select_data_subset_by_indices(test_indices, mode='test')
        torch.save(test_data, join(fold_folder, 'test_data.pt'))

    logger.info("Saving patient numbers")
    dataset_preparer.saver.save_patient_nums(train_data, val_data, folder=fold_folder)

    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)
    if len(test_indices) > 0:
        test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes)
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
        test_dataset=test_dataset,
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
    

if __name__ == '__main__':

    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    copy_pretrain_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()

    indices = list(range(len(data.pids)))

    if cfg.data.get('test_split', None) is not None:
        test_indices = random.sample(indices, int(len(indices)*cfg.data.test_split))
        test_indices_set = set(test_indices)
        indices = [i for i in indices if i not in test_indices_set]
    else:
        test_indices = []
    
    for fold, (train_indices, val_indices) in enumerate(get_n_splits_cv(data, N_SPLITS, indices)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        finetune_fold(cfg, data, train_indices, val_indices, fold, test_indices)
        
    compute_and_save_scores_mean_std(N_SPLITS, finetune_folder, mode='val')
    if len(test_indices) > 0:
        compute_and_save_scores_mean_std(N_SPLITS, finetune_folder, mode='test')

    
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
