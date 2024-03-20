import os
from datetime import datetime
from os.path import abspath, dirname, join, split
from pathlib import Path

import pandas as pd
import torch

from ehr2vec.common.azure import AzurePathContext, save_to_blobstore
from ehr2vec.common.config import Config, load_config
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.setup import get_args, setup_logger
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.evaluation.encodings import EHRTester
from ehr2vec.evaluation.utils import save_data

CONFIG_NAME = 'finetune_evaluate.yaml'
BLOBSTORE='PHAIR'

args = get_args(CONFIG_NAME)
config_path = join(dirname(abspath(__file__)), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def test_fold(cfg, finetune_folder: str, test_folder: str, fold:int, test_data: Data=None, run=None, logger=None)->None:
    """Finetune model on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    logger.info("Saving test pids")
    torch.save(test_data.pids, join(test_folder, 'test_pids.pt'))
    test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes)
    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config() 
    logger.info('Load best finetuned model to compute test scores')
    model = modelmanager.initialize_finetune_model(checkpoint, test_dataset)

    tester = EHRTester( 
        model=model, 
        test_dataset=None, # test only after training
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        test_folder=test_folder,
        mode=f'test_fold{fold}'
    )
    
    tester.model = model
    tester.test_dataset = test_dataset
    tester.evaluate(modelmanager.get_epoch(), mode='test')

def cv_test_loop(test_data: Data, finetune_folder: str, test_folder: str, 
                 n_splits:int, cfg=None, logger=None, run=None)->None:
    """Loop over cross validation folds. Save test results in test folder."""
    for fold in range(n_splits):
        fold += 1
        logger.info(f"Testing fold {fold}/{n_splits}")
        test_fold(cfg, finetune_folder, test_folder, fold, test_data, run, logger)

def compute_and_save_scores_mean_std(n_splits:int, test_folder: str, mode='test', logger=None)->None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    logger.info(f"Compute mean and std of {mode} scores")
    scores = []
    for fold in range(1, n_splits+1):
        table_path = join(test_folder, f'{mode}_fold{fold}_scores.csv')
        fold_scores = pd.read_csv(table_path)
        scores.append(fold_scores)
    scores = pd.concat(scores)
    scores_mean_std = scores.groupby('metric')['value'].agg(['mean', 'std'])
    scores_mean_std.to_csv(join(test_folder, f'{mode}_scores_mean_std.csv'))

def initialize_configuration_finetune(config_path:str, dataset_name:str=BLOBSTORE):
    """Load and adjust the configuration."""
    cfg = load_config(config_path)
    azure_context = AzurePathContext(cfg, dataset_name=dataset_name)
    cfg, run, mount_context = azure_context.azure_finetune_setup()
    if cfg.env=='azure':
        cfg.paths.output_path = 'outputs'
    else:
        cfg.paths.output_path = cfg.paths.model_path
    return cfg, run, mount_context, azure_context

def remove_tmp_prefixes(path:str)->Path:
    path_obj = Path(path)
    start_index = 1 # first part is \\
    for part in path_obj.parts:
        if part.startswith('tmp'):
            start_index += 1
    modified_parts = path_obj.parts[start_index:]
    return Path(*modified_parts)

def remove_tmp_prefixes_from_path_cfg(path_cfg:Config)->Config:
    """Strip prepend present in azure finetune configs for paths, starting with /tmp/tmp.../actual/path."""
    for key, value in path_cfg.items():
        if 'tmp' in value:
            path_cfg[key] = remove_tmp_prefixes(value)
    return path_cfg

def fix_cfg_for_azure(cfg:Config, azure_context: AzurePathContext)->Config:
    """Fix paths in config for azure.
    The saved finetune configs have /tmp/tmp.../actual/path
    after removing it, we need to prepend the new mounted path
    to every path in the config."""
    if cfg.env=='azure':
        cfg.paths = remove_tmp_prefixes_from_path_cfg(cfg.paths) 
        azure_context.cfg = cfg 
        cfg, _, _ = azure_context.azure_finetune_setup() 
    return cfg

def update_config(cfg:Config, finetune_folder:str)->Config:
    """Update config with pretrain and ft information."""
    finetune_config = load_config(join(finetune_folder, 'finetune_config.yaml'))
    pretrain_config = load_config(join(finetune_folder, 'pretrain_config.yaml'))
    cfg.data.update(finetune_config.data)
    cfg.outcome = finetune_config.outcome
    cfg.model = finetune_config.model
    cfg.trainer_args = finetune_config.trainer_args

    cfg.data.update(pretrain_config.data)
    cfg.paths.update(finetune_config.paths)
    cfg.model.update(pretrain_config.model)
    return cfg

def log_config(cfg, logger):
    for key, value in cfg.items():
        logger.info(f"{key}: {value}")

def main():
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)
    
    # create test folder
    date = datetime.now().strftime("%Y%m%d-%H%M")
    test_folder = join(cfg.paths.output_path, f'test_{date}')
    os.makedirs(test_folder, exist_ok=True)

    finetune_folder = cfg.paths.get("model_path")
    logger = setup_logger(test_folder, 'test_info.log')
    logger.info(f"Config Paths: {cfg.paths}")
    logger.info(f"Update config with pretrain and ft information.")
    cfg = update_config(cfg, finetune_folder)
    cfg = fix_cfg_for_azure(cfg, azure_context)
    cfg.save_to_yaml(join(test_folder, 'evaluate_config.yaml'))
    logger.info(f"Config Paths after fix: {cfg.paths}")

    fold_dirs = [fold_folder for fold_folder in os.listdir(finetune_folder) if fold_folder.startswith('fold_')]
    n_splits = len(fold_dirs)
    log_config(cfg, logger)
    cfg.paths.run_name = split(test_folder)[-1]
    dataset_preparer = DatasetPreparer(cfg)
    data = dataset_preparer.prepare_finetune_data()    
    if 'predefined_pids' in cfg.paths:
        logger.info(f"Load test pids from {cfg.paths.predefined_pids}")
        test_pids = torch.load(join(cfg.paths.predefined_pids, 'test_pids.pt')) 
        if len(test_pids)!=len(set(test_pids)):
            logger.warn(f'Test pids contain duplicates. Test pids len {len(test_pids)}, unique pids {len(set(test_pids))}.')
            logger.info('Removing duplicates')
            test_pids = list(set(test_pids))
        test_data = data.select_data_subset_by_pids(test_pids, mode='test')
    else:
        logger.info(f"Use all data for testing.")
        test_data = data
    save_data(test_data, test_folder)
    cv_test_loop(test_data, finetune_folder, test_folder, n_splits, cfg, logger, run)
    compute_and_save_scores_mean_std(n_splits, test_folder, mode='test', logger=logger)    
    
    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, remove_tmp_prefixes(cfg.paths.model_path)))
        mount_context.stop()
    logger.info('Done')


if __name__ == '__main__':
    main()
