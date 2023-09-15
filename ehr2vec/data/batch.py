import logging
import os
from os.path import join
from typing import List, Tuple

import numpy as np
import torch
from common.logger import TqdmToLogger
from common.utils import check_directory_for_features
from tqdm import tqdm

logger = logging.getLogger(__name__)  # Get the logger for this module
class DataSet:
    def __init__(self):
        self.pids = None
        self.file_ids = None

class Batches:
    def __init__(self, cfg, pids: List[List[str]]):
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
        
        torch.save(self.train.pids, join(self.cfg.output_dir, 'train_pids.pt'))
        torch.save(self.val.pids, join(self.cfg.output_dir, 'val_pids.pt'))
        torch.save(self.test.pids, join(self.cfg.output_dir, 'test_pids.pt'))
        torch.save(self.train.file_ids, join(self.cfg.output_dir, 'train_file_ids.pt'))
        torch.save(self.val.file_ids, join(self.cfg.output_dir, 'val_file_ids.pt'))
        torch.save(self.test.file_ids, join(self.cfg.output_dir, 'test_file_ids.pt'))
    
    def split_batches(self)-> None:
        """Splits the batches into train, validation and test sets"""
        file_ids = np.arange(len(self.pids))
        np.random.shuffle(file_ids)
        # calculate the number of batches for each set
        val_end = int(self.split_ratios['val'] * len(file_ids))
        train_end = val_end + int(self.split_ratios['train'] * len(file_ids))
        if self.split_ratios['test'] == 0:
            train_end = len(file_ids)
        # split the batches into train, validation and test
        
        self.val.file_ids = file_ids[:val_end]
        self.train.file_ids = file_ids[val_end:train_end]
        self.test.file_ids = file_ids[train_end:]
        
    def split_pids(self)-> None:
        """Splits the pids into train, validation and test sets"""
        self.train.pids, self.val.pids, self.test.pids = self.get_pids('train'), self.get_pids('val'), self.get_pids('test')
    
    def get_pids(self, set_: str)-> List[str]:
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
    
    def load_splits(self, data_dir)-> None:
        self.train.pids = torch.load(join(data_dir, 'train_pids.pt'))
        self.val.pids = torch.load(join(data_dir, 'val_pids.pt'))
        self.test.pids = torch.load(join(data_dir, 'test_pids.pt'))
        self.train.file_ids = torch.load(join(data_dir, 'train_file_ids.pt'))
        self.val.file_ids = torch.load(join(data_dir, 'val_file_ids.pt'))
        self.test.file_ids = torch.load(join(data_dir, 'test_file_ids.pt'))

    @staticmethod
    def flatten(ls_of_ls: List[List])-> List:
        return [item for sublist in ls_of_ls for item in sublist] 

class BatchTokenize:
    def __init__(self, tokenizer, cfg, tokenized_dir_name='tokenized'):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tokenized_dir_name = tokenized_dir_name
        os.makedirs(join(cfg.output_dir, self.tokenized_dir_name), exist_ok=True)
        self.pids = {}
    def tokenize(self, batches: Batches)-> Tuple[List[str]]:
        train_files = self.batch_tokenize(batches.train.file_ids, mode='train')
        self.tokenizer.freeze_vocabulary()
        self.tokenizer.save_vocab(join(self.cfg.output_dir, self.tokenized_dir_name, 'vocabulary.pt'))
        val_files = self.batch_tokenize(batches.val.file_ids, mode='val')
        test_files = self.batch_tokenize(batches.test.file_ids, mode='test')
        return train_files, val_files, test_files

    def batch_tokenize(self, batches, mode='train'):
        """Tokenizes batches and saves them"""
        if check_directory_for_features(self.cfg.loader.data_dir):
            features_dir = join(self.cfg.loader.data_dir, 'features')
        else:
            features_dir = join(self.cfg.output_dir, 'features')
        encoded = {}
        pids = []
        for batch in tqdm(batches, desc=f'Tokenizing {mode} batches', file=TqdmToLogger(logger)):
            features = torch.load(join(features_dir, f'features_{batch}.pt'))
            pids_batch = torch.load(join(features_dir, f'pids_features_{batch}.pt'))
            encoded_batch = self.tokenizer(features)
            self.merge_dicts(encoded, encoded_batch)
            pids = pids + pids_batch 
        torch.save(encoded, join(self.cfg.output_dir, self.tokenized_dir_name, f'tokenized_{mode}.pt'))
        torch.save(pids, join(self.cfg.output_dir, self.tokenized_dir_name, f'pids_{mode}.pt'))
        assert len(pids) == len(encoded['concept']), f"Length of pids ({len(pids)}) does not match length of encoded ({len(encoded['concept'])})"
        
    @staticmethod
    def merge_dicts(dict1:dict, dict2:dict):
        """Merges two dictionaries"""
        for key, value in dict2.items():
            dict1.setdefault(key, []).extend(value)
        