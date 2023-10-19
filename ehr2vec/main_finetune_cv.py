import os
from os.path import abspath, dirname, join
from typing import Iterator, List, Tuple

import pandas as pd
import torch
from common.config import instantiate, load_config
from common.loader import DatasetPreparer, Loader
from common.setup import (add_pretrain_info_to_cfg, adjust_paths_for_finetune,
                          azure_finetune_setup, copy_data_config, get_args,
                          setup_run_folder)
from common.utils import Data
from data.dataset import BinaryOutcomeDataset
from evaluation.utils import get_pos_weight, get_sampler
from model.model import BertForFineTuning
from sklearn.model_selection import KFold
from torch.optim import AdamW
from trainer.trainer import EHRTrainer

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 2  # You can change this to desired value

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
    dataset_preparer.saver.save_patient_nums(train_data, val_data, folder=fold_folder)

    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)

    logger.info('Initializing model')
    model = Loader(cfg).load_model(BertForFineTuning, 
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
        scheduler = instantiate(cfg.scheduler, **{'optimizer': optimizer})

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
    cfg = add_pretrain_info_to_cfg(cfg, mount_context)
    return cfg, run, mount_context, model_path


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

    cfg, run, mount_context, model_path = initialize_configuration()
    logger, finetune_folder = setup_run_folder(cfg)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    copy_data_config(cfg, finetune_folder)

    logger.info("Prepare Features")
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    
    for fold, (train_indices, val_indices) in enumerate(get_n_splits(data, N_SPLITS)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        finetune_fold(data, train_indices, val_indices, fold)
        
    compute_validation_scores_mean_std(finetune_folder)
    
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
