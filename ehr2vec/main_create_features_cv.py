"""
Creates features for pt, ft crossvalidation.
Adds patients used for model selection to pretrain set.
Excludes special patients (COVID).
"""
import os
import shutil
from os.path import join

import torch
from common.azure import AzurePathContext, save_to_blobstore
from common.config import load_config
from common.logger import TqdmToLogger
from common.setup import DirectoryPreparer, get_args
from common.utils import check_directory_for_features
from data.batch import Batches, BatchTokenize
from data.concept_loader import ConceptLoaderLarge
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
from tqdm import tqdm

CONFIG_NAME = 'data_pretrain.yaml'
BLOBSTORE = 'PHAIR'

args = get_args(CONFIG_NAME, 'data_pretrain')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_data(config_path):
    """
        Loads data
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    cfg = load_config(config_path)
    cfg, _, mount_context = AzurePathContext(cfg).azure_data_pretrain_setup()

    logger = DirectoryPreparer(config_path).prepare_directory(cfg)  
    logger.info('Mount Dataset')
    
    logger.info('Initialize Processors')
    logger.info('Starting feature creation and processing')
    if not check_directory_for_features(cfg.loader.data_dir):
        pids = create_and_save_features(ConceptLoaderLarge(**cfg.loader), 
                                        Handler(**cfg.handler), 
                                        Excluder(**cfg.excluder), 
                                        cfg, logger)
        torch.save(pids, join(cfg.output_dir, 'features', 'pids_features.pt'))
    else:
        pids = torch.load(join(cfg.loader.data_dir, 'features', 'pids_features.pt'))
  
    check_and_clear_directory(cfg, logger, tokenized_dir_name=cfg.get('tokenized_dir_name','tokenized'))
    batches = Batches(cfg, pids)
    folds = batches.split_batches_cv()
    logger.info('Tokenizing')
    tokenized_dir = join(cfg.output_dir, cfg.get('tokenized_dir_name','tokenized'))
    for i, fold in enumerate(folds):
        tokenizer = EHRTokenizer(config=cfg.tokenizer)
        batch_tokenize = BatchTokenize(pids, tokenizer, cfg, tokenized_dir_name=cfg.get('tokenized_dir_name','tokenized'))
        fold_dir = join(tokenized_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        batch_tokenize.batch_tokenize(fold['pretrain'], save_dir=fold_dir)
        batch_tokenize.tokenizer.freeze_vocabulary()
        batch_tokenize.tokenizer.save_vocab(join(fold_dir, 'vocabulary.pt'))
        batch_tokenize.batch_tokenize(fold['finetune_test'], save_dir=fold_dir)
    shutil.copy(config_path, join(tokenized_dir, 'data_cfg.yaml'))
    if cfg.env=='azure':
        save_to_blobstore(local_path=cfg.run_name, 
                          remote_path=join(BLOBSTORE, 'features', cfg.run_name))
        mount_context.stop()
    logger.info('Finished')

def check_and_clear_directory(cfg, logger, tokenized_dir_name='tokenized'):
    tokenized_dir = join(cfg.output_dir, tokenized_dir_name)
    tokenized_files = os.listdir(tokenized_dir) 
    if len(tokenized_files)>0:
        logger.warning(f"The directory {tokenized_dir} is not empty.")
        logger.warning(f"Deleting tokenized files.")
        for file in tokenized_files:
            file_path = join(tokenized_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)



def create_and_save_features(conceptloader, handler, excluder, cfg, logger, )-> list:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    pids = []
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Batch Process Data', file=TqdmToLogger(logger))):
        feature_maker = FeatureMaker(cfg.features) # Otherwise appended to old features
        features_batch, pids_batch = feature_maker(concept_batch, patient_batch)
        features_batch = handler(features_batch)
        features_batch, _, kept_indices  = excluder(features_batch)
        kept_pids = [pids_batch[idx] for idx in kept_indices]
        torch.save(features_batch, join(cfg.output_dir, 'features', f'features_{i}.pt'))
        torch.save(kept_pids, join(cfg.output_dir, 'features', f'pids_features_{i}.pt'))
        pids.append(kept_pids)
    return pids

if __name__ == '__main__':
    main_data(config_path)


