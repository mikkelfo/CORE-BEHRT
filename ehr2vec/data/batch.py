import logging
import os
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
from common.loader import load_assigned_pids, load_exclude_pids
from common.logger import TqdmToLogger
from common.utils import check_directory_for_features
from tqdm import tqdm

logger = logging.getLogger(__name__)  # Get the logger for this module
PRETRAIN = 'pretrain'
FINETUNE = 'finetune'
TEST = 'test'

@dataclass
class Split:
    pids: list = None
    mode: str = None

class Batches:
    """Class for splitting batches into pretrain, finetune and test sets"""
    def __init__(self, cfg, pids: List[List[str]]):
        """Initializes the class and splits the batches into pretrain, finetune and test sets
        pids should contain all the pids in the dataset, including assigned pids.
        Assigned pids should be a dictionary with the key being the split name and the value being a list of pids"""
        self.cfg = cfg

        self.predefined_splits_dir = cfg.get('predefined_splits_dir', None)
        self.flattened_pids = self.flatten(pids)
        logger.info(f"Total number of pids: {len(self.flattened_pids)}")
        if self.predefined_splits_dir is None:
            exclude_pids = load_exclude_pids(cfg)
            logger.info(f"Number of pids to exclude: {len(exclude_pids)}")
            assigned_pids = load_assigned_pids(cfg)
            for split, split_assigned_pids in assigned_pids.items():
                logger.info(f"Number of assigned pids in split {split}: {len(split_assigned_pids)}")
    
            # We exclude all the assigned pids from the flattened pids, and assign them to the splits later
            assigned_pids_set = set(self.flatten([v for v in assigned_pids.values()]))
            pids_to_exclude_set = set(exclude_pids).union(assigned_pids_set)
            self.flattened_pids = [pid for pid in self.flattened_pids if pid not in pids_to_exclude_set]
            self.assigned_pids = assigned_pids

            self.split_ratios = cfg['split_ratios']
            assert sum(self.split_ratios.values()) == 1, f"Sum of split ratios must be 1. Current sum: {sum(self.split_ratios.values())}"

    def split_batches(self)-> Dict[str, Split]:
        """Splits the batches into pretrain, finetune and test sets. """
        if self.predefined_splits_dir is not None:
            logger.warn(f'Loading predefined splits from {self.predefined_splits_dir}. Ignores all other settings related to splits.')
            return self.get_predefined_splits()
        np.random.seed(42)
        np.random.shuffle(self.flattened_pids)
        # calculate the number of batches for each set
        finetune_end, pretrain_end = self.calculate_split_indices(len(self.flattened_pids))
        # split the batches into pretrain, finetune and test
        splits = {
            PRETRAIN: self.flattened_pids[finetune_end:pretrain_end],
            FINETUNE: self.flattened_pids[:finetune_end],
        }
        if TEST in self.split_ratios:
            splits[TEST]= self.flattened_pids[pretrain_end:]
            
        for split, pids in self.assigned_pids.items():
            if split in splits:
                splits[split].extend(pids)
            else:
                raise ValueError(f"Split name {split} not recognized. Must be one of {splits.keys()}")
        split_dic = {}
        for split, pids in splits.items():
            logger.info(f"Final number of pids in split {split}: {len(pids)}")
            split_dic[split] = Split(pids=pids, mode=split)
        return split_dic
    
    def get_predefined_splits(self)-> Dict[str, Split]:
        """Loads predefined splits from predefined_splits_dir if in config."""
        split_files = {file.split('_')[1].split('.')[0 ]:file for file in os.listdir(self.predefined_splits_dir) if file.startswith('pids_')}
        assert len(split_files)>0, f"No predefined splits found in {self.predefined_splits_dir}"
        logger.info(f"Loading splits {split_files.keys()}")
        splits = {}
        all_predefined_pids_set = set()
        for mode, file in split_files.items():
            pids = torch.load(join(self.predefined_splits_dir, file))
            all_predefined_pids_set.update(set(pids))
            splits[mode] = Split(pids=pids, mode=mode)
        assert all_predefined_pids_set.issubset(set(self.flattened_pids)), f"Predefined pids are not a subset of all pids."
        return splits

    def create_split(self, indices: List, mode: str)-> Split:
        """Create a Split object for the given indices and mode. And assigns pids."""
        pids = [self.flattened_pids[i] for i in indices]
        pids += self.assigned_pids.get(mode, [])
        return Split(pids=pids, mode=mode)

    def calculate_split_indices(self, total_length: int)-> Tuple[int, int]:
        """Calculates the indices for each split based on configured ratios."""
        finetune_end = int(self.split_ratios[FINETUNE] * total_length)
        pretrain_end = finetune_end + int(self.split_ratios[PRETRAIN] * total_length)

        if self.split_ratios.get(TEST, 0) == 0:
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

    def create_tokenized_directory(self)->None:
        """Creates the directory for storing tokenized data."""
        os.makedirs(join(self.cfg.output_dir, self.tokenized_dir_name), exist_ok=True)
    @staticmethod
    def map_pids_to_file_ids(pids: List[List[str]])->Dict[str, int]:
        """Maps PIDs to file IDs."""
        return {pid: fileid for fileid, sublist in enumerate(pids) for pid in sublist}

    def tokenize(self, splits: Dict[str, Split])->None:
        """Tokenizes all batches. The returned order is according to the pids in splits."""
        self.batch_tokenize(splits[PRETRAIN])
        self.tokenizer.freeze_vocabulary()
        self.save_vocabulary()
        self.batch_tokenize(splits[FINETUNE])
        if TEST in splits:
            if len(splits[TEST].pids) > 0:
                self.batch_tokenize(splits[TEST])
    
    def save_vocabulary(self)->None:
        """Saves the tokenizer's vocabulary."""
        self.tokenizer.save_vocab(join(self.cfg.output_dir, self.tokenized_dir_name, 'vocabulary.pt'))
        
    def batch_tokenize(self, split: Split, save_dir=None)->None:
        """Tokenizes batches and saves them"""
        features_dir = self.get_features_directory()
        encoded, pids = self.tokenize_split(split, features_dir)
        self.save_tokenized_data(encoded, pids, split.mode, save_dir=save_dir)
        
    def tokenize_split(self, split: Split, features_dir: str)->None:    
        """
        Loops through all files to get pids in split and tokenizes them. 
        Returns tokenized features and pids ordered according to split.pids.
        """
        encoded, pids = {}, []
        # we need to know which pid is in which file
        split_pids_set = set(split.pids)
        pid2fileid = {pid: file_id for pid, file_id in self.pid2fileid.items() if pid in split_pids_set} 
        fileid2pid = self.invert_dictionary(pid2fileid)
        # select only file ids 
        for file_id, selected_pids_in_file in tqdm(fileid2pid.items(), desc=f'Tokenizing {split.mode} batches', file=TqdmToLogger(logger)):
            filtered_features, filtered_pids = self.load_and_filter_batch(file_id, selected_pids_in_file, features_dir)
            encoded_batch = self.tokenizer(filtered_features)
            self.merge_dicts(encoded, encoded_batch)
            pids = pids + filtered_pids

        # use the order of split.pids to ensure the order of encoded and pids is the same
        assert set(split.pids)==set(pids), f"Split pids ({len(split.pids)}) and pids ({len(pids)}) do not match"
        pids, encoded = self.reorder_pids_and_encoded_feats(split.pids, pids, encoded)
        
        assert len(pids) == len(encoded['concept']), f"Length of pids ({len(pids)}) does not match length of encoded ({len(encoded['concept'])})"
        
        return encoded, pids
    @staticmethod
    def reorder_pids_and_encoded_feats(split_pids: List[str], pids: List[str], encoded: Dict[str, torch.tensor])->Tuple[List[str], Dict[str, torch.tensor]]:
        """Reorders pids and encoded to match the order of split_pids"""
        indices = [pids.index(pid) for pid in split_pids]
        for key, value in encoded.items():
            encoded[key] = [value[idx] for idx in indices]
        return split_pids, encoded
    
    @staticmethod
    def load_and_filter_batch(file_id: str, 
                            selected_pids_in_file: List[str], 
                            features_dir: str)->Tuple[Dict[str, torch.tensor], List[str]]:
        """Load features and pids for file_id, filter them by selected_pids_in_file and tokenize them."""
        features = torch.load(join(features_dir, f'features_{file_id}.pt'))
        pids_file = torch.load(join(features_dir, f'pids_features_{file_id}.pt'))
        filtered_features, filtered_pids = BatchTokenize.filter_features_by_pids(features, pids_file, selected_pids_in_file)
        return filtered_features, filtered_pids 
    
    def save_tokenized_data(self, encoded: Dict[str, torch.tensor], pids: List[str], mode:str, save_dir:str=None)->None:
        if save_dir is None:
            save_dir = join(self.cfg.output_dir, self.tokenized_dir_name)
        torch.save(encoded, join(save_dir, f'tokenized_{mode}.pt'))
        torch.save(pids, join(save_dir, f'pids_{mode}.pt'))
        
    def get_features_directory(self)->str:
        if check_directory_for_features(self.cfg.loader.data_dir):
            return join(self.cfg.loader.data_dir, 'features')
        else:
            return join(self.cfg.output_dir, 'features')
        
    @staticmethod  
    def filter_features_by_pids(features: Dict[str, List[List]], pids_batch: List[str], split_pids: List[str])->Tuple[Dict[str, List], List[str]]:
        """Filters features and pids. Keep only split_pids by pids"""
        filtered_features = {}
        assert set(split_pids).issubset(set(pids_batch)), f"Batch pids are not a subset of pids in file. Batch pids: {split_pids}, pids in file: {pids_file}"
        indices_to_keep = [pids_batch.index(pid) for pid in split_pids]
        kept_pids = [pids_batch[idx] for idx in indices_to_keep]
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