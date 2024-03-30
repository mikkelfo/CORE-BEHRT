"""
Getting the following statistics for train_val, test and combined set:
Number of patients, number of positive patients
sequence lengths, age at censoring, trajectory length (arrays, mean+-std)
"""
import os
from os.path import abspath, dirname, join, split

import pandas as pd
import torch

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import Initializer
from ehr2vec.common.loader import load_and_select_splits
from ehr2vec.common.setup import (DirectoryPreparer, copy_data_config,
                                  copy_pretrain_config, get_args)
from ehr2vec.common.utils import Data
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.data.split import get_n_splits_cv
from ehr2vec.data.utils import Utilities
from ehr2vec.evaluation.stats import (calculate_statistics, plot_and_save_hist,
                                      save_gender_distribution)
from ehr2vec.evaluation.utils import (
    check_data_for_overlap, save_data,
    split_into_test_data_and_train_val_indices)

CONFIG_NAME = 'finetune_stats.yaml'
BLOBSTORE='PHAIR'
N_SPLITS = 2

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def save_split_fold(train_data:Data, val_data:Data, 
                fold:int, test_data: Data=None)->None:
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    os.makedirs(fold_folder, exist_ok=True)
    logger.info("Saving pids")
    torch.save(train_data.pids, join(fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(fold_folder, 'val_pids.pt'))
    if len(test_data) > 0:
        torch.save(test_data.pids, join(fold_folder, 'test_pids.pt'))

def process_and_save(data: Data, func: callable, name: str, split:str, folder: str)->tuple:
    tensor_data = torch.tensor(func(data)).float()
    positive_indices = [i for i, outcome in enumerate(data.outcomes) if pd.notna(outcome)]
    
    plot_and_save_hist(tensor_data, name, split, folder, positive_indices)
    torch.save(tensor_data, join(folder, f'{split}_{name}.pt'))
    
    stats_all = calculate_statistics(tensor_data)
    stats_pos = calculate_statistics(tensor_data[positive_indices])

    return stats_all, stats_pos

def save_stats(finetune_folder:str, train_val_data:Data, test_data: Data=None)->None:
    """Save basic stats"""
    data_dict = {'train_val':train_val_data, 'test':test_data}
    logger.info("Saving pids")
    torch.save(train_val_data.pids, join(finetune_folder, 'train_val_pids.pt'))
    if len(test_data) > 0:
        torch.save(test_data.pids, join(finetune_folder, 'test_pids.pt'))
    logger.info("Saving patient numbers")
    dataset_preparer.saver.save_patient_nums_general(data_dict, folder=finetune_folder)
    save_gender_distribution(data_dict, finetune_folder)
    logger.info("Saving sequence lengths, age at censoring and trajectory lengths")
    # Initialize DataFrames for storing statistics
    stats, positive_stats = {}, {}
    
    # List of metric calculation functions and their names
    metric_functions = [
        (Utilities.calculate_ages_at_censor_date, 'age'),
        (Utilities.calculate_sequence_lengths, 'sequence_len'),
        (Utilities.calculate_trajectory_lengths, 'trajectory_len')
    ]

    for split, data in data_dict.items():
        for func, name in metric_functions:
            all_stats, pos_stats = process_and_save(data, func, name, split, finetune_folder)
            # Store all data statistics
            for i, measure in enumerate(['mean', 'std', 'median', 'lower_quartile', 'upper_quartile']):
                stat_name = f'{name}_{measure}'
                stats.setdefault(stat_name, {})[split] = all_stats[i]
                positive_stats.setdefault(stat_name, {})[split] = pos_stats[i] 
    
    # Convert dictionaries to DataFrames
    stats = pd.DataFrame(stats).transpose()
    positive_stats = pd.DataFrame(positive_stats).transpose()
    # Save DataFrames to CSV
    stats.to_csv(join(finetune_folder, 'patient_stats.csv'), index=True)
    positive_stats.to_csv(join(finetune_folder, 'patient_stats_positives.csv'), index=True)

def _limit_train_patients(indices_or_pids: list)->list:
    if 'number_of_train_patients' in cfg.data:
        if len(indices_or_pids) >= cfg.data.number_of_train_patients:
            indices_or_pids = indices_or_pids[:cfg.data.number_of_train_patients]
            logger.info(f"Number of train patients is limited to {cfg.data.number_of_train_patients}")
        else:
            raise ValueError(f"Number of train patients is {len(indices_or_pids)}, but should be at least {cfg.data.number_of_train_patients}")
    return indices_or_pids

def split_and_save(data: Data, train_indices: list, val_indices: list, fold: int, test_data: Data=None):
    train_data = data.select_data_subset_by_indices(train_indices, mode='train')
    val_data = data.select_data_subset_by_indices(val_indices, mode='val')
    check_data_for_overlap(train_data, val_data, test_data)
    save_split_fold(train_data, val_data, fold, test_data)

def cv_loop_split_and_save(data: Data, train_val_indices: list, test_data: Data)->None:
    """Loop over cross validation folds."""
    for fold, (train_indices, val_indices) in enumerate(get_n_splits_cv(data, N_SPLITS, train_val_indices)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        logger.info("Splitting data")
        train_indices = _limit_train_patients(train_indices)
        split_and_save(data, train_indices, val_indices, fold, test_data)

def cv_get_predefined_splits(data: Data, predefined_splits_dir: str, test_data: Data)->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    N_SPLITS = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Loading fold {fold}/{len(fold_dirs)}")
        train_data, val_data = load_and_select_splits(fold_dir, data)
        train_pids = _limit_train_patients(train_data.pids)
        train_data = data.select_data_subset_by_pids(train_pids, mode='train')
        check_data_for_overlap(train_data, val_data, test_data)
        save_split_fold(train_data, val_data, fold, test_data)
    return N_SPLITS, train_data, val_data


if __name__ == '__main__':
    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    copy_pretrain_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_data()    
    if 'abspos' not in data.features:
        raise ValueError('No absolute positions found in data, needed to calculate stats. Check remove features tag in config and make sure BehrtAdapter is not activated as it removes abspos.')
    logger.info('Splitting data')

    if 'predefined_splits' in cfg.paths:
        logger.info('Using predefined splits')
        test_pids = torch.load(join(cfg.paths.predefined_splits, 'test_pids.pt')) if os.path.exists(join(cfg.paths.predefined_splits, 'test_pids.pt')) else []
        test_pids = list(set(test_pids))
        test_data = data.select_data_subset_by_pids(test_pids, mode='test')
        save_data(test_data, finetune_folder)
        N_SPLITS, train_data, val_data  = cv_get_predefined_splits(
            data, cfg.paths.predefined_splits, test_data)
        train_val_pids = train_data.pids + val_data.pids
        train_val_data = data.select_data_subset_by_pids(train_val_pids, mode='train_val')
    else:
        test_data, train_val_indices = split_into_test_data_and_train_val_indices(cfg, data)
        train_val_data = data.select_data_subset_by_indices(train_val_indices, mode='train_val')
        check_data_for_overlap(train_val_data, test_data)
        cv_loop_split_and_save(data, train_val_indices, test_data)
    
    save_data(test_data, finetune_folder)
    save_stats(finetune_folder, train_val_data, test_data)
    
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
