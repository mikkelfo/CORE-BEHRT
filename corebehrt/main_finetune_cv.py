import os
from os.path import abspath, dirname, join, split

import torch

from corebehrt.common.azure import save_to_blobstore
from corebehrt.common.initialize import Initializer, ModelManager
from corebehrt.common.loader import load_and_select_splits
from corebehrt.common.setup import (DirectoryPreparer, copy_data_config,
                                  copy_pretrain_config, get_args)
from corebehrt.common.utils import Data, compute_number_of_warmup_steps
from corebehrt.data.dataset import BinaryOutcomeDataset
from corebehrt.data.prepare_data import DatasetPreparer
from corebehrt.data.split import get_n_splits_cv
from corebehrt.evaluation.utils import (compute_and_save_scores_mean_std, save_data,
    split_into_test_data_and_train_val_indices)
from corebehrt.trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 2  # You can change this to desired value
BLOBSTORE='PHAIR'
DEAFAULT_VAL_SPLIT = 0.2

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def finetune_fold(cfg, train_data:Data, val_data:Data, 
                fold:int, test_data: Data=None)->None:
    """Finetune model on one fold"""
    if 'scheduler' in cfg:
        logger.info('Computing number of warmup steps')
        compute_number_of_warmup_steps(cfg, len(train_data))
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
    modelmanager.load_model_config() 
    model = modelmanager.initialize_finetune_model(checkpoint, train_dataset)
    
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(model, train_dataset)
    epoch = modelmanager.get_epoch()


    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        test_dataset=None, # test only after training
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

    logger.info('Load best finetuned model to compute test scores')
    modelmanager_trained = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager_trained.load_checkpoint()
    modelmanager.load_model_config() 
    model = modelmanager_trained.initialize_finetune_model(checkpoint, train_dataset)
    trainer.model = model
    trainer.test_dataset = test_dataset
    trainer._evaluate(checkpoint['epoch'], mode='test')



def split_and_finetune(data: Data, train_indices: list, val_indices: list, fold: int, test_data: Data=None):
    train_data = data.select_data_subset_by_indices(train_indices, mode='train')
    val_data = data.select_data_subset_by_indices(val_indices, mode='val')
    finetune_fold(cfg, train_data, val_data, fold, test_data)

def _limit_patients(indices_or_pids: list, split: str)->list:
    if f'number_of_{split}_patients' in cfg.data:
        number_of_patients = cfg.data.get(f'number_of_{split}_patients')
        if len(indices_or_pids) >= number_of_patients:
            indices_or_pids = indices_or_pids[:number_of_patients]
            logger.info(f"Number of {split} patients is limited to {number_of_patients}")
        else:
            raise ValueError(f"Number of train patients is {len(indices_or_pids)}, but should be at least {number_of_patients}")
    return indices_or_pids

def cv_loop(data: Data, train_val_indices: list, test_data: Data)->None:
    """Loop over cross validation folds."""
    for fold, (train_indices, val_indices) in enumerate(get_n_splits_cv(data, N_SPLITS, train_val_indices)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        logger.info("Splitting data")
        train_indices = _limit_patients(train_indices, 'train')
        val_indices = _limit_patients(val_indices, 'val')
        split_and_finetune(data, train_indices, val_indices, fold, test_data)

def finetune_without_cv(data: Data, train_val_indices:list, test_data: Data=None)->None:
    val_split = cfg.data.get('val_split', DEAFAULT_VAL_SPLIT)
    logger.info(f"Splitting train_val of length {len(train_val_indices)} into train and val with val_split={val_split}")
    train_indices = train_val_indices[:int(len(train_val_indices)*(1-val_split))]
    val_indices = train_val_indices[int(len(train_val_indices)*(1-val_split)):]
    split_and_finetune(data, train_indices, val_indices, 1, test_data)

def cv_loop_predefined_splits(data: Data, predefined_splits_dir: str, test_data: Data)->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    N_SPLITS = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        train_data, val_data = load_and_select_splits(fold_dir, data)
        train_pids = _limit_patients(train_data.pids, 'train')
        val_pids = _limit_patients(val_data.pids, 'val')
        if len(train_pids)<len(train_data.pids):
            train_data = data.select_data_subset_by_pids(train_pids, mode='train')
        if len(val_pids)<len(val_data.pids):
            val_data = data.select_data_subset_by_pids(val_pids, mode='val')
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
        test_pids = list(set(test_pids))
        test_data = data.select_data_subset_by_pids(test_pids, mode='test')
        save_data(test_data, finetune_folder)
        N_SPLITS = cv_loop_predefined_splits(data, cfg.paths.predefined_splits, test_data)

    else:
        logger.info('Splitting data')
        test_data, train_val_indices = split_into_test_data_and_train_val_indices(cfg, data)
        save_data(test_data, finetune_folder)
        if N_SPLITS > 1:
            cv_loop(data, train_val_indices, test_data)
        else:
            finetune_without_cv(data, train_val_indices, test_data)

    compute_and_save_scores_mean_std(N_SPLITS, finetune_folder, mode='val')
    if len(test_data) > 0:
        compute_and_save_scores_mean_std(N_SPLITS, finetune_folder, mode='test')    
    
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
