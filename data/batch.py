from os.path import join

import numpy as np
import torch
from tqdm import tqdm
from common.logger import TqdmToLogger


class DataSet:
    def __init__(self):
        self.pids = None
        self.file_ids = None

class Batches:
    def __init__(self, cfg, pids: list[list[str]]):
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
    def __init__(self, tokenizer, cfg, logger):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.logger = logger

    def tokenize(self, batches: Batches)-> tuple[list[str]]:
        train_files = self.batch_tokenize(batches.train.file_ids, mode='train')
        self.tokenizer.freeze_vocabulary()
        self.tokenizer.save_vocab(join(self.cfg.output_dir, 'vocabulary.pt'))
        val_files = self.batch_tokenize(batches.val.file_ids, mode='val')
        test_files = self.batch_tokenize(batches.test.file_ids, mode='test')
        return train_files, val_files, test_files

    def batch_tokenize(self, batches, mode='train'):
        """Tokenizes batches and saves them"""
        files = []
        for batch in tqdm(batches, desc=f'Tokenizing {mode} batches', file=TqdmToLogger(self.logger)):
            features = torch.load(join(self.cfg.output_dir, 'features', f'features_{batch}.pt'))
            train_encoded = self.tokenizer(features)
            torch.save(train_encoded, join(self.cfg.output_dir, 'tokenized', f'tokenized_{mode}_{batch}.pt'))
            files.append(f'tokenized_{mode}_{batch}.pt')
        return files