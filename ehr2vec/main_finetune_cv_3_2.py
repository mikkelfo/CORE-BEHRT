"""
Finetune model on 3/5 of the data, validate on 1/5 and test on 1/5. 
This results in 10 runs with 5 different test sets. 
"""

import math
import os
from collections import Counter, defaultdict
from os.path import abspath, dirname, join
from typing import List

import pandas as pd
import torch
from common.azure import save_to_blobstore
from common.initialize import Initializer, ModelManager
from common.setup import (DirectoryPreparer, copy_data_config,
                          copy_pretrain_config, get_args)
from common.utils import Data
from data.dataset import BinaryOutcomeDataset
from data.prepare_data import DatasetPreparer
from data.split import get_n_splits_cv_k_over_n
from evaluation.utils import compute_and_save_scores_mean_std
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
# When changing N_SPLITS or TRAIN_SPLITS, make sure that N_SPLITS >= TRAIN_SPLITS and adjust MAX_SET_REPETITIONS accordingly.
N_SPLITS = 5  
TRAIN_SPLITS = 3
MAX_SET_REPETITIONS = 2
BLOBSTORE='PHAIR'

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def finetune_fold(cfg, data:Data, train_indices:List[int], val_indices:List[int], 
                  test_indices:List[int], test_set_id: int, fold:int):
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Splitting data")
    train_data = data.select_data_subset_by_indices(train_indices, mode='train')
    val_data = data.select_data_subset_by_indices(val_indices, mode='val')
    test_data = data.select_data_subset_by_indices(test_indices, mode='test')

    logger.info("Saving pids")
    torch.save(train_data.pids, join(fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(fold_folder, 'val_pids.pt'))
    torch.save(test_data.pids, join(fold_folder, 'test_pids.pt'))

    logger.info("Saving patient numbers")
    dataset_preparer.saver.save_patient_nums(train_data, val_data, folder=fold_folder)
    
    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)
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

def select_val_test_sets(test_val_sets_ids:List[int], test_val_pids_sets:List[List[int]], val_set_id_counter:Counter, test_set_id_counter:Counter):
    """Select val and test sets, such that each set is used equally often as val and test set."""
    if  (val_set_id_counter[test_val_sets_ids[0]] == MAX_SET_REPETITIONS) or (test_set_id_counter[test_val_sets_ids[1]] == MAX_SET_REPETITIONS):
        val_set_id = test_val_sets_ids[1]
        test_set_id = test_val_sets_ids[0]
        val_pids = test_val_pids_sets[1]
        test_pids = test_val_pids_sets[0]
    else:
        val_set_id = test_val_sets_ids[0]
        test_set_id = test_val_sets_ids[1]
        val_pids = test_val_pids_sets[0]
        test_pids = test_val_pids_sets[1]        
    
    val_set_id_counter[val_set_id] += 1
    test_set_id_counter[test_set_id] += 1
    return val_pids, val_set_id, test_pids, test_set_id

if __name__ == '__main__':
    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    copy_pretrain_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    fold2sets = defaultdict(list)
    test_set_id_counter = Counter()
    val_set_id_counter = Counter()
    for fold, (train_pids, test_val_pids_sets, test_val_sets_ids) in enumerate(get_n_splits_cv_k_over_n(data, N_SPLITS, TRAIN_SPLITS)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        
        val_pids, val_set_id, test_pids, test_set_id = select_val_test_sets(test_val_sets_ids, test_val_pids_sets, val_set_id_counter, test_set_id_counter)
        logger.info(f"Val set id: {val_set_id}")
        logger.info(f"Test set id: {test_set_id}")
        fold2sets['fold'].append(fold)
        fold2sets['val_set_id'].append(val_set_id)
        fold2sets['test_set_id'].append(test_set_id)
        finetune_fold(cfg, data, train_pids, val_pids, test_pids, test_set_id, fold)
    pd.DataFrame(fold2sets).to_csv(join(finetune_folder, 'test_sets_in_folds.csv'), index=False)
    n_folds = math.comb(N_SPLITS, TRAIN_SPLITS)
    compute_and_save_scores_mean_std(n_folds, finetune_folder, mode='test')
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
