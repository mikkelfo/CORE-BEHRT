import logging
import os
import random
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from corebehrt.common.config import Config
from corebehrt.common.loader import load_assigned_pids, load_exclude_pids
from corebehrt.common.logger import TqdmToLogger
from corebehrt.common.utils import check_directory_for_features
from corebehrt.data.utils import Utilities

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
    def __init__(self, cfg: Config, pids: List[List[str]]):
        """Initializes the class and splits the batches into pretrain, finetune and test sets
        pids should contain all the pids in the dataset, including assigned pids.
        Assigned pids should be a dictionary with the key being the split name and the value being a list of pids"""

        self.predefined_splits_dir = cfg.get('predefined_splits_dir', None)
        self.flattened_pids = self.flatten(pids)
        logger.info(f"Total number of pids: {len(self.flattened_pids)}")
        if self.predefined_splits_dir is None:
            exclude_pids = load_exclude_pids(cfg)
            logger.info(f"Number of pids to exclude: {len(exclude_pids)}")
            assigned_pids = load_assigned_pids(cfg)
            assigned_pids = self.filter_assigned_pids(assigned_pids, self.flattened_pids)
            for split, split_assigned_pids in assigned_pids.items():
                logger.info(f"Number of assigned pids in split {split}: {len(split_assigned_pids)}")
    
            # We exclude all the assigned pids from the flattened pids, and assign them to the splits later
            assigned_pids_set = set(self.flatten([v for v in assigned_pids.values()]))
            pids_to_exclude_set = set(exclude_pids).union(assigned_pids_set)
            self.flattened_pids = [pid for pid in self.flattened_pids if pid not in pids_to_exclude_set]
            self.assigned_pids = assigned_pids

            self.split_ratios = cfg['split_ratios']
            assert round(sum(self.split_ratios.values()), 5) == 1, f"Sum of split ratios must be 1. Current sum: {sum(self.split_ratios.values())}"

    def split_batches(self)-> Dict[str, Split]:
        """Splits the batches into pretrain, finetune and test sets. """
        if self.predefined_splits_dir is not None:
            logger.warn(f'Loading predefined splits from {self.predefined_splits_dir}. Ignores all other settings related to splits.')
            return self.get_predefined_splits()
        
        # Calculate the remaining number of pids for each split
        total_pids = len(self.flattened_pids) # total number of pids, excluding the assigned pids

        for pids in self.assigned_pids.values():
            total_pids += len(pids) # include the assigned pids in the total number of pids to calculate the remaining ratios

        self.update_split_ratios(total_pids)
        self.shuffle_pids()

        # calculate the number of batches for each set
        finetune_end, pretrain_end = self.calculate_split_indices(len(self.flattened_pids))
        splits = self.allocate_splits(pretrain_end, finetune_end)
        
        return self.create_split_dict(splits)
    
    def get_predefined_splits(self)-> Dict[str, Split]:
        """Loads predefined splits from predefined_splits_dir if in config."""
        split_files = {file.split('_')[1].split('.')[0]: file for file in os.listdir(self.predefined_splits_dir) if file.startswith('pids_')}
        assert len(split_files)>0, f"No predefined splits found in {self.predefined_splits_dir}"
        logger.info(f"Loading splits {split_files.keys()}")
        splits = {}
        all_predefined_pids_set = set()
        for mode, file in split_files.items():
            pids = torch.load(join(self.predefined_splits_dir, file))
            if not set(pids).issubset(set(self.flattened_pids)):
                diff = len(set(pids).difference(set(self.flattened_pids)))
                logger.warning(f"Predefined pids in mode {mode} contain pids that are not in the dataset. {diff}")
                pids = list(set(pids).intersection(set(self.flattened_pids)))
            assert len(pids)==len(set(pids)), f"Predefined pids for split {mode} contain duplicates."
            all_predefined_pids_set.update(set(pids))
            splits[mode] = Split(pids=pids, mode=mode)
        assert all_predefined_pids_set.issubset(set(self.flattened_pids)), f"Predefined pids are not a subset of all pids."

        return splits
    
    def update_split_ratios(self, total_length: int) -> None:
        """
        Calculates the remaining split ratios after allocating assigned pids,
        and updates the split ratios. 
        Raises an error if the ratios cannot be satisfied.
        """
        # Calculate the total number of pids already allocated to each split
        allocated_counts = {split: len(pids) for split, pids in self.assigned_pids.items()}
        # Check if the allocated counts exceed the split ratios
        for split, ratio in self.split_ratios.items():
            # Calculate the maximum number of pids that can be allocated to this split
            max_allowed_pids = int(ratio * total_length)
            if allocated_counts.get(split, 0) > max_allowed_pids:
                raise ValueError(f"Cannot satisfy split ratios for split '{split}'. Allocated count exceeds the ratio limit {int(ratio * total_length)}.")

        # Calculate the remaining length after allocating assigned pids
        remaining_length = total_length - sum(allocated_counts.values())
        # Calculate remaining ratios for the unallocated pids
        remaining_ratios = {}
        for split, ratio in self.split_ratios.items():
            # Calculate the portion of the total dataset that should be allocated to this split
            # after accounting for already allocated pids
            target_ratio = ratio * total_length - allocated_counts.get(split, 0)
            remaining_ratios[split] = target_ratio / remaining_length if remaining_length > 0 else 0
        self.split_ratios = remaining_ratios
        assert round(sum(self.split_ratios.values()), 5) == 1, f"Sum of split ratios must be 1. Current sum: {sum(self.split_ratios.values())}"

    def shuffle_pids(self):
        """Shuffles the flattened pids."""
        random.seed(42)
        random.shuffle(self.flattened_pids)

    @staticmethod
    def create_split_dict(splits: Dict[str, Split])-> Dict[str, Split]:
        split_dic = {}
        for split, pids in splits.items():
            logger.info(f"Final number of pids in split {split}: {len(pids)}")
            split_dic[split] = Split(pids=pids, mode=split)
        return split_dic

    def allocate_splits(self, pretrain_end: int, finetune_end: int)-> Dict[str, list]:
        """Allocates pids to each split."""
        # split the batches into pretrain, finetune and test
        splits = {
            PRETRAIN: self.flattened_pids[finetune_end:pretrain_end],
            FINETUNE: self.flattened_pids[:finetune_end],
        }
        if TEST in self.split_ratios:
            splits[TEST]= self.flattened_pids[pretrain_end:]
        logger.info("Pids in each split before assigning pids:")
        for split, pids in splits.items():
            logger.info(f"Number of pids in split {split}: {len(pids)}")
        for split, pids in self.assigned_pids.items():
            if split in splits:
                splits[split].extend(pids)
            else:
                raise ValueError(f"Split name {split} not recognized. Must be one of {splits.keys()}")
        return splits
    
    def calculate_split_indices(self, total_length: int)-> Tuple[int, int]:
        """Calculates the indices for each split based on configured ratios."""
        finetune_end = int(self.split_ratios[FINETUNE] * total_length)
        pretrain_end = finetune_end + int(self.split_ratios[PRETRAIN] * total_length)

        if self.split_ratios.get(TEST, 0) == 0:
            pretrain_end = total_length
        return finetune_end, pretrain_end

    @staticmethod
    def flatten(ls_of_ls: List[List])-> List:
        """Flattens a list of lists."""
        return [item for sublist in ls_of_ls for item in sublist] 
    @staticmethod
    def filter_assigned_pids(assigned_pids: Dict[str, List[str]], pids: List[str])->Dict[str, List[str]]:
        """Filters assigned pids by pids."""
        pids_set = set(pids)
        return {split: [pid for pid in pids if pid in pids_set] for split, pids in assigned_pids.items()}

class BatchTokenize:
    def __init__(self, pids: List[List[str]], tokenizer, cfg: Config, tokenized_dir_name:str='tokenized'):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.tokenized_dir_name = tokenized_dir_name
        # Create the tokenized directory
        os.makedirs(join(self.cfg.output_dir, self.tokenized_dir_name), exist_ok=True)

        self.pid2fileid = self.map_pids_to_file_ids(pids)
        
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
        if TEST in splits and len(splits[TEST].pids) > 0:
            self.batch_tokenize(splits[TEST])
    
    def save_vocabulary(self)->None:
        """Saves the tokenizer's vocabulary."""
        self.tokenizer.save_vocab(join(self.cfg.output_dir, self.tokenized_dir_name, 'vocabulary.pt'))
        
    def batch_tokenize(self, split: Split, save_dir=None)->None:
        """Tokenizes batches and saves them"""
        features_dir = self.get_features_directory()
        encoded, pids = self.tokenize_split(split, features_dir)
        self.save_tokenized_data(encoded, pids, split.mode, save_dir=save_dir)
        return encoded, pids
        
    def tokenize_split(self, split: Split, features_dir: str)->None:    
        """
        Loops through all files to get pids in split and tokenizes them. 
        Returns tokenized features and pids ordered according to split.pids.
        """
        encoded, pids = {}, []
        # we need to know which pid is in which file
        split_pids_set = set(split.pids)
        assert split_pids_set.issubset(set(self.pid2fileid.keys())), f"Split pids ({len(split_pids_set)}) is not a subset of pid2fileid keys ({len(self.pid2fileid.keys())})"
        pid2fileid = {pid: file_id for pid, file_id in self.pid2fileid.items() if pid in split_pids_set} 
        fileid2pid = self.invert_dictionary(pid2fileid)
        # select only file ids 
        for file_id, selected_pids_in_file in tqdm(fileid2pid.items(), desc=f'Tokenizing {split.mode} batches', file=TqdmToLogger(logger)):
            filtered_features, filtered_pids = self.load_and_filter_batch(file_id, selected_pids_in_file, features_dir)
            encoded_batch = self.tokenizer(filtered_features)
            self.merge_dicts(encoded, encoded_batch)
            pids.extend(filtered_pids)

        encoded, pids = Utilities.select_and_reorder_feats_and_pids(encoded, pids, split.pids)
        
        return encoded, pids
    
    @staticmethod
    def load_and_filter_batch(file_id: str, 
                            selected_pids_in_file: List[str], 
                            features_dir: str)->Tuple[Dict[str, torch.tensor], List[str]]:
        """Load features and pids for file_id, filter them by selected_pids_in_file and tokenize them."""
        features = torch.load(join(features_dir, f'features_{file_id}.pt'))
        pids_file = torch.load(join(features_dir, f'pids_features_{file_id}.pt'))
        filtered_features, filtered_pids = Utilities.select_and_reorder_feats_and_pids(
            features, pids_file, selected_pids_in_file)
        return filtered_features, filtered_pids 
    
    def save_tokenized_data(self, encoded: Dict[str, torch.tensor], pids: List[str], mode:str, save_dir:str=None)->None:
        """Saves the tokenized data to disk."""
        if save_dir is None:
            save_dir = join(self.cfg.output_dir, self.tokenized_dir_name)
        torch.save(encoded, join(save_dir, f'tokenized_{mode}.pt'))
        torch.save(pids, join(save_dir, f'pids_{mode}.pt'))
        
    def get_features_directory(self)->str:
        """Returns the directory where features are stored."""
        if check_directory_for_features(self.cfg.loader.data_dir):
            return join(self.cfg.loader.data_dir, 'features')
        else:
            return join(self.cfg.output_dir, 'features')
    @staticmethod
    def merge_dicts(dict1:dict, dict2:dict)->None:
        """Merges two dictionaries in place (dict1)"""
        for key, finetune in dict2.items():
            dict1.setdefault(key, []).extend(finetune)

    @staticmethod    
    def invert_dictionary(original_dict: Dict[str, str])->Dict[str, List[str]]:
        """Inverts a dictionary, stores values as keys and keys as values. New values are stored in lists."""
        inverted_dict = {}
        for key, value in original_dict.items():
            if value not in inverted_dict:
                inverted_dict[value] = []
            inverted_dict[value].append(key)
        return inverted_dict