import logging
import os
from os.path import join
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import torch
from common.logger import TqdmToLogger
from common.utils import check_directory_for_features
from tqdm import tqdm
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)  # Get the logger for this module


@dataclass
class Split:
    pids: list = None
    mode: str = None

class Batches:
    """Class for splitting batches into pretrain, finetune and test sets"""
    def __init__(self, cfg, pids: List[List[str]], exclude_pids: List[str]=[], assigned_pids: Dict[str, List[str]]={}):
        """Initializes the class and splits the batches into pretrain, finetune and test sets
        pids should contain all the pids in the dataset, including assigned pids.
        Assigned pids should be a dictionary with the key being the split name and the value being a list of pids"""
        self.cfg = cfg

        flattened_pids = self.flatten(pids)
        logger.info(f"Total number of pids: {len(flattened_pids)}")
        # We exclude all the assigned pids from the flattened pids, and assign them to the splits later
        assigned_pids_set = set(self.flatten([v for v in assigned_pids.values()]))
        logger.info(f"Total Number of assigned pids: {len(assigned_pids_set)}")
        logger.info(f"Total number of pids to exclude: {len(exclude_pids)}")
        pids_to_exclude = set(exclude_pids).union(assigned_pids_set)
        self.flattened_pids = [pid for pid in flattened_pids if pid not in pids_to_exclude]
        self.assigned_pids = assigned_pids

        self.split_ratios = cfg['split_ratios']
        assert sum(self.split_ratios.values()) == 1, f"Sum of split ratios must be 1. Current sum: {sum(self.split_ratios.values())}"
        self.n_splits = cfg.get('n_splits', 2)

    def split_batches(self)-> Dict[str, Split]:
        """Splits the batches into pretrain, finetune and test sets"""
        np.random.shuffle(self.flattened_pids)
        # calculate the number of batches for each set
        finetune_end, pretrain_end = self.calculate_split_indices(len(self.flattened_pids))
        # split the batches into pretrain, finetune and test
        splits = {
            'pretrain': self.flattened_pids[finetune_end:pretrain_end],
            'finetune': self.flattened_pids[:finetune_end],
        }
        if 'test' in self.split_ratios:
            splits['test']= self.flattened_pids[pretrain_end:]
            
        for split_name, pids in self.assigned_pids.items():
            if split_name in splits:
                splits[split_name].extend(pids)
            else:
                raise ValueError(f"Split name {split_name} not recognized. Must be one of ['pretrain', 'finetune', 'test', 'finetune_test']")
        return {name: Split(pids=pids, mode=name) for name, pids in splits.items()}

    def split_batches_cv(self)-> List[Dict[str, Split]]:
        """Splits the batches into pretrain and finetune_test (finetune and test can be split during cv.) sets for crossvalidation"""
        logger.info(f"Number of splits: {self.n_splits}")
        kfold = KFold(n_splits=self.n_splits, shuffle=True)
        
        folds = []
        for pretrain_indices, finetune_test_indices in kfold.split(self.flattened_pids):
            pretrain_split = self.create_split(pretrain_indices, 'pretrain')
            ft_test_split = self.create_split(finetune_test_indices, 'finetune_test')
            folds.append({'pretrain': pretrain_split, 'finetune_test': ft_test_split})          
            logger.info(f"Pretrain split: {len(pretrain_split.pids)}, Finetune test split: {len(ft_test_split.pids)}")
        return folds

    def create_split(self, indices, mode):
        """Create a Split object for the given indices and mode. And assigns pids."""
        pids = [self.flattened_pids[i] for i in indices]
        pids += self.assigned_pids.get(mode, [])
        return Split(pids=pids, mode=mode)

    def calculate_split_indices(self, total_length):
        """Calculates the indices for each split based on configured ratios."""
        finetune_end = int(self.split_ratios['finetune'] * total_length)
        pretrain_end = finetune_end + int(self.split_ratios['pretrain'] * total_length)

        if self.split_ratios.get('test', 0) == 0:
            pretrain_end = total_length
        return finetune_end, pretrain_end

    @staticmethod
    def flatten(ls_of_ls: List[List])-> List:
        return [item for sublist in ls_of_ls for item in sublist] 

class BatchTokenize:
    def __init__(self, pids: List[List[str]], tokenizer, cfg, tokenized_dir_name:str='tokenized'):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tokenized_dir_name = tokenized_dir_name
        self.create_tokenized_directory()
        self.pid2fileid = self.map_pids_to_file_ids(pids)

    def create_tokenized_directory(self):
        """Creates the directory for storing tokenized data."""
        os.makedirs(join(self.cfg.output_dir, self.tokenized_dir_name), exist_ok=True)

    def map_pids_to_file_ids(self, pids):
        """Maps PIDs to file IDs."""
        return {pid: fileid for fileid, sublist in enumerate(pids) for pid in sublist}

    def tokenize(self, splits: Dict[str, Split]):
        """Tokenizes all batches."""
        self.batch_tokenize(splits['pretrain'])
        self.tokenizer.freeze_vocabulary()
        self.save_vocabulary()
        self.batch_tokenize(splits['finetune'])
        if 'test' in splits:
            if len(splits['test'].pids) > 0:
                self.batch_tokenize(splits['test'])
    
    def save_vocabulary(self):
        """Saves the tokenizer's vocabulary."""
        self.tokenizer.save_vocab(join(self.cfg.output_dir, self.tokenized_dir_name, 'vocabulary.pt'))
        
    def batch_tokenize(self, split: Split, save_dir=None)->None:
        """Tokenizes batches and saves them"""
        features_dir = self.get_features_directory()
        encoded, pids = self.tokenize_split(split, features_dir)
        self.save_tokenized_data(encoded, pids, split.mode, save_dir=save_dir)
        
    def tokenize_split(self, split: Split, features_dir: str)->None:    
        encoded = {}
        pids = []
        split_pids = set(split.pids)
        pid2fileid = {pid: file_id for pid, file_id in self.pid2fileid.items() if pid in split_pids}
        fileid2pid = self.invert_dictionary(pid2fileid)
        # select only file ids 
        for file_id, selected_pids_in_file in tqdm(fileid2pid.items(), desc=f'Tokenizing {split.mode} batches', file=TqdmToLogger(logger)):
            features = torch.load(join(features_dir, f'features_{file_id}.pt'))
            pids_file = torch.load(join(features_dir, f'pids_features_{file_id}.pt'))
            
            filtered_features, filtered_pids = self.filter_features_by_pids(features, pids_file, selected_pids_in_file)
            
            encoded_batch = self.tokenizer(filtered_features)
            self.merge_dicts(encoded, encoded_batch)
            pids = pids + filtered_pids
        
        assert len(pids) == len(encoded['concept']), f"Length of pids ({len(pids)}) does not match length of encoded ({len(encoded['concept'])})"
        
        return encoded, pids
        
    def save_tokenized_data(self, encoded: Dict[str, torch.tensor], pids: List[str], mode:str, save_dir:str=None)->None:
        if save_dir is None:
            save_dir = join(self.cfg.output_dir, self.tokenized_dir_name)
        torch.save(encoded, join(save_dir, f'tokenized_{mode}.pt'))
        torch.save(pids, join(save_dir, f'pids_{mode}.pt'))
        
    def get_features_directory(self):
        if check_directory_for_features(self.cfg.loader.data_dir):
            return join(self.cfg.loader.data_dir, 'features')
        else:
            return join(self.cfg.output_dir, 'features')
        
    @staticmethod  
    def filter_features_by_pids(features: dict, pids_file: List[str], split_pids: List[str])->dict:
        """Filters features and pids. Keep only split_pids by pids"""
        filtered_features = {}
        assert set(split_pids).issubset(set(pids_file)), f"Batch pids are not a subset of pids in file. Batch pids: {split_pids}, pids in file: {pids_file}"
        
        indices_to_keep = set([pids_file.index(pid) for pid in split_pids])
        kept_pids = [pids_file[idx] for idx in indices_to_keep]
        for key, feature in features.items():
            filtered_features[key] = [entry for idx, entry in enumerate(feature) if idx in indices_to_keep]
        return filtered_features, kept_pids 
    
    @staticmethod
    def merge_dicts(dict1:dict, dict2:dict)->None:
        """Merges two dictionaries in place (dict1)"""
        for key, finetuneue in dict2.items():
            dict1.setdefault(key, []).extend(finetuneue)
    @staticmethod    
    def invert_dictionary(original_dict: Dict[str, str])->Dict[str, List[str]]:
        """Inverts a dictionary, stores values as keys and keys as values. New values are stored in lists."""
        inverted_dict = {}
        for key, value in original_dict.items():
            if value not in inverted_dict:
                inverted_dict[value] = []
            inverted_dict[value].append(key)
        return inverted_dict