import os
from os.path import join
from shutil import copyfile

import torch
from tqdm import tqdm

# from data.batch import 
from data.config import load_config, Config
from data.concept_loader import ConceptLoader
from data.dataset import MLMLargeDataset
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
from data_fixes.infer import Inferrer

import numpy as np  

config_path = join("configs", "data.yaml")
cfg = load_config(config_path)

def prepare_directory(cfg: Config):
    """Creates output directory and copies config file"""
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    copyfile(config_path, join(cfg.output_dir, 'data_config.yaml'))

    
class DataSet:
    def __init__(self):
        self.pids = None
        self.file_ids = None

class Batches:
    def __init__(self, cfg: Config, pids: list[list[str]]):
        self.pids = pids
        self.cfg = cfg
        self.split_ratios = cfg.split_ratios
        self.train = DataSet()
        self.val = DataSet()
        self.test = DataSet()

    def split_and_save(self)-> None:
        """
        Splits batches into train, val, test and saves them
        """
        self.split_batches()
        self.split_pids()
        
        torch.save(self.train.pids, join(cfg.output_dir, 'train_pids.pt'))
        torch.save(self.val.pids, join(cfg.output_dir, 'val_pids.pt'))
        torch.save(self.test.pids, join(cfg.output_dir, 'test_pids.pt'))
    
    def split_batches(self)-> None:
        """Splits the batches into train, validation and test sets"""
        file_ids = np.arange(len(self.pids))
        np.random.shuffle(file_ids)
        # calculate the number of batches for each set
        train_end = int(self.split_ratios['train'] * len(file_ids))
        val_end = train_end + int(self.split_ratios['val'] * len(file_ids))
        # split the batches into train, validation and test
        
        self.train.file_ids = file_ids[:train_end]
        self.val.file_ids = file_ids[train_end:val_end]
        self.test.file_ids = file_ids[val_end:]

    def split_pids(self)-> None:
        """Splits the pids into train, validation and test sets"""
        self.train.pids, self.val.pids, self.test.pids = self.get_pids('train'), self.get_pids('val'), self.get_pids('test')
    
    def get_pids(self, set_: str)-> list[str]:
        """Returns the pids for the given indices"""
        if set_ == 'train':
            file_ids = self.train.file_ids
        elif set_ == 'val':
            file_ids = self.val.file_ids    
        elif set_ == 'test':
            file_ids = self.test.file_ids
        else:
            raise ValueError(f'Invalid set {set_}')
        return self.flatten([self.pids[i] for i in file_ids])
        
    @staticmethod
    def flatten(ls_of_ls: list[list])-> list:
        return [item for sublist in ls_of_ls for item in sublist] 

class BatchTokenize:
    def __init__(self, tokenizer, cfg: Config):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def tokenize(self, batches: Batches)-> tuple[list[str]]:
        print('Tokenizing')
        train_files = self.batch_tokenize(batches.train.file_ids, mode='train')
        self.tokenizer.freeze_vocabulary()
        self.tokenizer.save_vocab(join(cfg.output_dir, 'vocabulary.pt'))
        val_files = self.batch_tokenize(batches.val.file_ids, mode='val')
        test_files = self.batch_tokenize(batches.test.file_ids, mode='test')
        return train_files, val_files, test_files

    def batch_tokenize(self, batches, mode='train'):
        """Tokenizes batches and saves them"""
        files = []
        for batch in tqdm(batches, desc=f'Tokenizing {mode} batches'):
            features = torch.load(join(cfg.output_dir, f'features_{batch}.pt'))
            train_encoded = self.tokenizer(features)
            torch.save(train_encoded, join(cfg.output_dir, f'encoded_{mode}_{batch}.pt'))
            files.append(join(self.cfg.output_dir, f'encoded_{mode}_{batch}.pt'))
        return files

def main_data(cfg):
    """
        Loads data
        Infers nans
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    prepare_directory(cfg)
    
    conceptloader = ConceptLoader(**cfg.loader)
    inferrer = Inferrer()
    # outcome_maker = OutcomeMaker(cfg)
    feature_maker = FeatureMaker(cfg)
    handler = Handler()
    excluder = Excluder(cfg)

    pids = []
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Process')):
        pids.append(patient_batch['PID'].tolist())
        # concept_batch = inferrer(concept_batch)
        # patient_batch = OutcomeMaker(cfg)(patient_batch)
        features_batch = feature_maker(concept_batch, patient_batch)
        # print(features_batch['concept'][0])
        # features_batch = handler(features_batch)
        # features_batch, _ = excluder(features_batch)
        torch.save(features_batch, join(cfg.output_dir, f'features_{i}.pt'))
    batches = Batches(cfg, pids)
    batches.split_and_save()

    # Tokenize
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    batch_tokenize = BatchTokenize(tokenizer, cfg)
    train_files, val_files, test_files = batch_tokenize.tokenize(batches)

    print('Saving datasets')
    cfg.dataset.vocabulary = tokenizer.vocabulary
    train_dataset = MLMLargeDataset(train_files, **{**cfg.dataset, 'num_patients': len(batches.train.pids)})
    test_dataset = MLMLargeDataset(test_files, **{**cfg.dataset, 'num_patients': len(batches.test.pids)})
    val_dataset = MLMLargeDataset(val_files, **{**cfg.dataset, 'num_patients': len(batches.val.pids)})
    torch.save(train_dataset, join(cfg.output_dir, 'train_dataset.pt'))
    torch.save(test_dataset, join(cfg.output_dir,'test_dataset.pt'))
    torch.save(val_dataset, join(cfg.output_dir,'val_dataset.pt'))

if __name__ == '__main__':
    main_data(cfg)

