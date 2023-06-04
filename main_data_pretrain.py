import os
from os.path import join
from shutil import copyfile

import torch
from tqdm import tqdm

from data import utils
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
cfg = utils.load_config(config_path)
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


def prepare_directory(cfg: Config):
    """Creates output directory and copies config file"""
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    copyfile(config_path, join(cfg.output_dir, 'data_config.yaml'))



class BatchTokenize:
    def __init__(self, tokenizer, cfg: Config):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def tokenize(self, batches)-> tuple[list[str]]:
        print('Tokenizing')
        train_batches, val_batches, test_batches = batches.train.batches, batches.val.batches, batches.test.batches
        train_files = self.batch_tokenize(cfg, self.tokenizer, train_batches, mode='train')
        self.tokenizer.freeze_vocabulary()
        self.tokenizer.save_vocab(join(cfg.output_dir, 'vocabulary.pt'))
        val_files = self.batch_tokenize(cfg, self.tokenizer, val_batches, mode='val')
        test_files = self.batch_tokenize(cfg, self.tokenizer, test_batches, mode='test')
        return train_files, val_files, test_files

    def batch_tokenize(self, batches, mode='train'):
        """Tokenizes batches and saves them"""
        files = []
        for batch in tqdm(batches, desc=f'Tokenizing {mode} batches'):
            features = torch.load(join(cfg.output_dir, f'features{batch}.pt'))
            train_encoded = self.tokenizer(features)
            torch.save(train_encoded, join(cfg.output_dir, f'encoded_{mode}{batch}.pt'))
            files.append(join(self.cfg.output_dir, f'encoded_{mode}{batch}.pt'))
        return files

class Batches:
    def __init__(self, cfg: Config, pids: list[list[str]]):
        self.pids = pids
        self.cfg = cfg
        self.split_ratios = cfg.split_ratios

    def split_and_save(self)-> dict:
        """
        Splits batches into train, val, test and saves them
        Returns: Dictionary with train, val, test batches and pids
        """
        train_batches, val_batches, test_batches = self.split_batches(self.pids, cfg.split_ratios)
        train_pids, val_pids, test_pids = self.split_pids(train_batches, val_batches, test_batches)
        
        self.train.pids = self.flatten(train_pids)
        self.val.pids = self.flatten(val_pids)
        self.test.pids = self.flatten(test_pids)
        self.train.batches = train_batches
        self.val.batches = val_batches
        self.test.batches = test_batches

        torch.save(train_pids, join(cfg.output_dir, 'train_pids.pt'))
        torch.save(val_pids, join(cfg.output_dir, 'val_pids.pt'))
        torch.save(test_pids, join(cfg.output_dir, 'test_pids.pt'))
    
    def split_batches(self):
        """Splits the batches into train, validation and test sets"""
        batches = np.arange(len(self.pids))
        np.random.shuffle(batches)
        # calculate the number of batches for each set
        train_end = int(self.split_ratios['train'] * len(batches))
        val_end = train_end + int(self.split_ratios['val'] * len(batches))
        # split the batches into train, validation and test
        
        train_batches = batches[:train_end]
        val_batches = batches[train_end:val_end]
        test_batches = batches[val_end:]

        return train_batches, val_batches, test_batches

    def split_pids(self, train_batches, val_batches, test_batches):
        """Splits the pids into train, validation and test sets"""
        return self.get_pids(train_batches), self.get_pids(val_batches), self.get_pids(test_batches)
    
    def get_pids(self, indices: list):
        """Returns the pids for the given indices"""
        return [self.pids[i] for i in indices]
    @staticmethod
    def flatten(ls_of_ls: list[list]):
        return [item for sublist in ls_of_ls for item in sublist] 

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
        torch.save(features_batch, join(cfg.output_dir, f'features{i}.pt'))
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

