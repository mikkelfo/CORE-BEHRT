"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from os.path import join

import torch
from common import azure
from common.config import load_config
from common.loader import check_directory_for_features
from common.logger import TqdmToLogger
from common.setup import prepare_directory, get_args
from data.batch import Batches, BatchTokenize
from data.concept_loader import ConceptLoaderLarge
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
from tqdm import tqdm

args = get_args('data_pretrain.yaml', 'data_pretrain')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def check_and_clear_directory(cfg, logger):
    tokenized_dir = join(cfg.output_dir, 'tokenized')
    tokenized_files = os.listdir(tokenized_dir) 
    if len(tokenized_files)>0:
        logger.warning(f"The directory {tokenized_dir} is not empty.")
        logger.warning(f"Deleting tokenized files.")
        for file in tokenized_files:
            os.remove(join(tokenized_dir, file))


def create_and_save_features(conceptloader, handler, excluder, cfg, logger, )-> list[list]:
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
    if cfg.env=='azure':
        _, mount_context = azure.setup_azure(cfg.run_name)
        mount_dir = mount_context.mount_point
        cfg.loader.data_dir = join(mount_dir, cfg.loader.data_dir) # specify paths here

    logger = prepare_directory(config_path, cfg)  
    logger.info('Mount Dataset')
    
    
    logger.info('Initialize Processors')
    logger.info('Starting feature creation and processing')
    if not check_directory_for_features(cfg.loader.data_dir, logger):
        pids = create_and_save_features(ConceptLoaderLarge(**cfg.loader), 
                                        Handler(**cfg.handler), 
                                        Excluder(**cfg.excluder), 
                                        cfg, logger)
        torch.save(pids, join(cfg.output_dir, 'features', 'pids_features.pt'))
    else:
        pids = torch.load(join(cfg.loader.data_dir, 'features', 'pids_features.pt'))
    logger.info('Finished feature creation and processing')
    
    logger.info('Splitting batches')
    batches = Batches(cfg, pids)
    batches.split_and_save()
    
    check_and_clear_directory(cfg, logger)
    logger.info('Tokenizing')
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    batch_tokenize = BatchTokenize(tokenizer, cfg, logger)
    batch_tokenize.tokenize(batches)
    
    logger.info('Saving file ids')
    torch.save(batches.train.file_ids, join(cfg.output_dir, 'train_file_ids.pt'))
    torch.save(batches.val.file_ids, join(cfg.output_dir, 'val_file_ids.pt'))
    torch.save(batches.test.file_ids, join(cfg.output_dir, 'test_file_ids.pt'))
    
    if cfg.env=='azure':
        from azure_run import file_dataset_save
        file_dataset_save(local_path=join('outputs', 'data'), datastore_name = "workspaceblobstore",
                    remote_path = join("PHAIR", "pretrain_dataset", cfg.run_name))
        mount_context.stop()
    logger.info('Finished')

if __name__ == '__main__':
    main_data(config_path)
