"""
Finetune model on 3/5 of the data, validate on 1/5 and test on 1/5. 
This results in 10 runs with 5 different test sets. 
"""

import os
from collections import defaultdict, namedtuple
from datetime import datetime
from os.path import abspath, dirname, join
from typing import List, Dict

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
from dataloader.collate_fn import dynamic_padding
from trainer.trainer import EHRTrainer
from trainer.utils import get_tqdm

CONFIG_NAME = 'finetune.yaml'
N_SPLITS = 5  # You can change this to desired value
TRAIN_SPLITS = 3
BLOBSTORE='PHAIR'

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def finetune_fold(cfg, data:Data, train_indices:List[int], val_indices:List[int], 
                  test_indices:List[int], test_set_id: int, fold:int, all_folds_metrics:dict):
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
    get_test_scores(cfg,train_dataset, test_dataset, test_set_id, fold, fold_folder,  
                    all_folds_metrics, trainer)

def get_test_scores(cfg, train_dataset:BinaryOutcomeDataset, 
                    test_dataset:BinaryOutcomeDataset, 
                    test_set_id:int, fold:int, fold_folder:str,
                    all_folds_metrics:dict, trainer):
    modelmanager = ModelManager(cfg, fold)
    checkpoint = modelmanager.load_checkpoint() 
    modelmanager.load_model_config() # check whether cfg is changed after that.
    model = modelmanager.initialize_finetune_model(checkpoint, train_dataset)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=cfg.trainer_args.batch_size, 
                                              shuffle=False, collate_fn=dynamic_padding)
    metric_values = validate(model, test_loader, trainer)
    torch.save(metric_values, join(fold_folder, f'test_scores_on_set_{test_set_id}.pt'))    
    all_folds_metrics[test_set_id].append(metric_values)

def validate(model, dataloader, trainer)->tuple:
    """Returns the validation loss and metrics"""
    model.eval()
    val_loop = get_tqdm(dataloader)
    val_loop.set_description('Post Hoc Validation')
    
    metric_values = {name: [] for name in trainer.metrics}
    logits_list = [] 
    targets_list = []

    with torch.no_grad():
        for batch in val_loop:
            trainer.batch_to_device(batch)
            outputs = model(batch)
            logits_list.append(outputs.logits.cpu())
            targets_list.append(batch['target'].cpu())
            
    metric_values = process_binary_classification_results(trainer.metrics, logits_list, targets_list)
    
    return metric_values

def process_binary_classification_results(metrics: dict, logits:list, targets:list)->dict:
        """Process results specifically for binary classification."""
        targets = torch.cat(targets)
        logits = torch.cat(logits)
        batch = {'target': targets}
        outputs = namedtuple('Outputs', ['logits'])(logits)
        acc_metrics = {}
        for name, func in metrics.items():
            v = func(outputs, batch)
            acc_metrics[name] = v
        return acc_metrics
from statistics import mean, stdev
def save_mean_and_std_test_scores(all_folds_test_scores: Dict[str, List[Dict]], finetune_folder: str)->None:
    """Compute mean and std of validation scores."""
    logger.info("Compute mean and std of validation scores")
    test_scores = []
    for _, fold_test_scores in all_folds_test_scores.items():
        for fold_test_score in fold_test_scores:
            test_scores.append(fold_test_score)
    # Calculate mean and std for each metric
    test_scores_summary = {metric: {'mean': mean([d[metric] for d in test_scores]),
                            'std': stdev([d[metric] for d in test_scores])}
                   for metric in test_scores[0]}
    
    test_scores_summary_df = pd.DataFrame.from_dict(test_scores_summary, orient='index')
    date = datetime.now().strftime("%Y%m%d-%H%M")
    test_scores_summary_df.to_csv(join(finetune_folder, f'test_scores_mean_std_{date}.csv'))

if __name__ == '__main__':

    cfg, run, mount_context, pretrain_model_path = Initializer.initialize_configuration_finetune(config_path)

    logger, finetune_folder = DirectoryPreparer.setup_run_folder(cfg)
    
    copy_data_config(cfg, finetune_folder)
    copy_pretrain_config(cfg, finetune_folder)
    cfg.save_to_yaml(join(finetune_folder, 'finetune_config.yaml'))
    
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_features()
    all_folds_test_scores = defaultdict(list)
    for fold, (train_pids, val_pids_sets, val_sets_ids) in enumerate(get_n_splits_cv_k_over_n(data, N_SPLITS, TRAIN_SPLITS)):
        fold += 1
        logger.info(f"Training fold {fold}/{N_SPLITS}")
        val_pids = val_pids_sets[0]
        test_pids = val_pids_sets[1]
        val_set_id = val_sets_ids[0]
        test_set_id = val_sets_ids[1]
        logger.info(f"Val set id: {val_set_id}")
        logger.info(f"Test set id: {test_set_id}")
        finetune_fold(cfg, data, train_pids, val_pids, test_pids, test_set_id, fold, all_folds_test_scores)
    torch.save(all_folds_test_scores, join(finetune_folder, 'all_folds_metrics.pt'))
    save_mean_and_std_test_scores(all_folds_test_scores, finetune_folder)
    
    if cfg.env=='azure':
        save_path = pretrain_model_path if cfg.paths.get("save_folder_path", None) is None else cfg.paths.save_folder_path
        save_to_blobstore(local_path=cfg.paths.run_name, 
                          remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done')
