from os.path import join
# from azureml.core import Dataset
import torch
from tqdm import tqdm

from common.config import load_config
from common.logger import TqdmToLogger
from common.setup import prepare_directory
from data.batch import Batches, BatchTokenize
from data.concept_loader import ConceptLoader
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
# from azure_run.run import Run
# from azure_run import datastore


# run = Run
# run.name(f"Pretrain base on med/diag")

# dataset = Dataset.File.from_files(path=(datastore(), 'PHAIR/formatted_data/diagnosis_medication'))

config_path = join('configs', 'data.yaml')

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
    logger = prepare_directory(config_path, cfg)  
    logger.info('Mount Dataset')
    
    # mount_context = dataset.mount()
    # mount_context.start()  # this will mount the file streams
    
    # cfg.loader.data_dir = mount_context.mount_point
    
    logger.info('Initialize Processors')
    conceptloader = ConceptLoader(**cfg.loader)
    
    handler = Handler(**cfg.handler)
    excluder = Excluder(**cfg.excluder)
    logger.info('Starting data processing')
    pids = []
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Batch Process Data', file=TqdmToLogger(logger))):
        pids.append(patient_batch['PID'].tolist())
        feature_maker = FeatureMaker(cfg.features) # Otherwise appended to old features
        features_batch = feature_maker(concept_batch, patient_batch)
        features_batch = handler(features_batch)
        features_batch, _, pids_batch  = excluder(features_batch)
        torch.save(features_batch, join(cfg.output_dir, 'features', f'features_{i}.pt'))
        torch.save(pids_batch, join(cfg.output_dir, 'features', f'pids_features_{i}.pt'))
        

    logger.info('Finished data processing')
    logger.info('Splitting batches')
    batches = Batches(cfg, pids)
    batches.split_and_save()

    logger.info('Tokenizing')
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    batch_tokenize = BatchTokenize(tokenizer, cfg, logger)
    batch_tokenize.tokenize(batches)
    
    logger.info('Saving file ids')
    torch.save(batches.train.file_ids, join(cfg.output_dir, 'train_file_ids.pt'))
    torch.save(batches.val.file_ids, join(cfg.output_dir, 'val_file_ids.pt'))
    torch.save(batches.test.file_ids, join(cfg.output_dir, 'test_file_ids.pt'))
    logger.info('Finished')
    
    # mount_context.stop()

if __name__ == '__main__':
    main_data(config_path)

