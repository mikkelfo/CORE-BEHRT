import os
import torch
from os.path import abspath, dirname, join, split

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import Initializer, ModelManager
from ehr2vec.common.setup import (DirectoryPreparer, copy_data_config,
                          copy_pretrain_config, get_args)
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.data.split import get_n_splits_cv
from ehr2vec.evaluation.utils import (check_data_for_overlap,
                              compute_and_save_scores_mean_std,
                              split_into_test_and_train_val_and_save_test_set)
from ehr2vec.trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 2  # You can change this to desired value
BLOBSTORE='PHAIR'

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def finetune_fold(cfg, train_data:Data, val_data:Data, 
                fold:int, test_data: Data=None)->None:
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    os.makedirs(fold_folder, exist_ok=True)
    os.makedirs(join(fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Saving patient numbers")
    logger.info("Saving pids")
    torch.save(train_data.pids, join(fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(fold_folder, 'val_pids.pt'))
    if len(test_data) > 0:
        torch.save(test_data.pids, join(fold_folder, 'test_pids.pt'))
    dataset_preparer.saver.save_patient_nums(train_data, val_data, folder=fold_folder)

    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)
    test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes) if len(test_data) > 0 else None
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

def limit_train_patients(indices_or_pids: list)->list:
    if 'number_of_train_patients' in cfg.data:
        if len(indices_or_pids) > cfg.data.number_of_train_patients:
            indices_or_pids = indices_or_pids[:cfg.data.number_of_train_patients]
            logger.info(f"Number of train patients is limited to {cfg.data.number_of_train_patients}")
        else:
            raise ValueError(f"Number of train patients is {len(indices_or_pids)}, but should be at least {cfg.data.number_of_train_patients}")
    return indices_or_pids

def cv_loop(data: Data, train_val_indices: list, test_data: Data)->None:
    """Loop over cross validation folds."""
    for fold, (train_indices, val_indices) in enumerate(get_n_splits_cv(data, N_SPLITS, train_val_indices)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        logger.info("Splitting data")
        train_indices = limit_train_patients(train_indices)
        train_data = data.select_data_subset_by_indices(train_indices, mode='train')
        val_data = data.select_data_subset_by_indices(val_indices, mode='val')
        check_data_for_overlap(train_data, val_data, test_data)
        finetune_fold(cfg, train_data, val_data, fold, test_data)

def cv_loop_predefined_splits(data: Data, predefined_splits_dir: str, test_data: Data)->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    N_SPLITS = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        logger.info("Load and select pids")
        train_pids = torch.load(join(fold_dir, 'train_pids.pt'))
        train_pids = limit_train_patients(train_pids)
        val_pids = torch.load(join(fold_dir, 'val_pids.pt'))
        train_data = data.select_data_subset_by_pids(train_pids, mode='train')
        val_data = data.select_data_subset_by_pids(val_pids, mode='val')
        check_data_for_overlap(train_data, val_data, test_data)
        finetune_fold(cfg, train_data, val_data, fold, test_data)
    return N_SPLITS

if __name__ == '__main__':
    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    copy_pretrain_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_data()    
    
    if 'predefined_splits' in cfg.paths:
        logger.info('Using predefined splits')
        test_pids = torch.load(join(cfg.paths.predefined_splits, 'test_pids.pt')) if os.path.exists(join(cfg.paths.predefined_splits, 'test_pids.pt')) else []
        test_data = data.select_data_subset_by_pids(test_pids, mode='test')
        if len(test_data) > 0:
            torch.save(test_data.pids, join(finetune_folder, 'test_pids.pt'))
        N_SPLITS = cv_loop_predefined_splits(data, cfg.paths.predefined_splits, test_data)

    else:
        logger.info('Splitting data')
        test_data, train_val_indices = split_into_test_and_train_val_and_save_test_set(cfg, data, finetune_folder)
        cv_loop(data, train_val_indices, test_data)
    
    compute_and_save_scores_mean_std(N_SPLITS, finetune_folder, mode='val')
    if len(test_data) > 0:
        compute_and_save_scores_mean_std(N_SPLITS, finetune_folder, mode='test')    
    
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
