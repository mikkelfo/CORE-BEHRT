import glob
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
import torch

from corebehrt.common.config import Config

logger = logging.getLogger(__name__)  # Get the logger for this module

def iter_patients(features: dict) -> Generator[dict, None, None]:
    """Iterate over patients in a features dict."""
    for i in range(len(features["concept"])):
        yield {key: values[i] for key, values in features.items()}

def check_patient_counts(concepts: pd.DataFrame, patients_info: pd.DataFrame, logger)->None:
    """Check that the number of patients in concepts and patients_info match."""
    if concepts.PID.nunique() != patients_info.PID.nunique():
            logger.warning(f"patients info contains {patients_info.PID.nunique()} patients != \
                        {concepts.PID.nunique()} unique patients in concepts")

def check_existing_splits(data_dir: str)-> bool:
    """Check if train, val, and test splits already exist in data_dir."""
    if os.path.exists(join(data_dir, 'train_pids.pt')) and\
        os.path.exists(join(data_dir, 'val_pids.pt')) and\
        os.path.exists(join(data_dir, 'test_pids.pt')) and\
        os.path.exists(join(data_dir, 'train_file_ids.pt')) and\
        os.path.exists(join(data_dir, 'val_file_ids.pt')) and\
        os.path.exists(join(data_dir, 'test_file_ids.pt')):
        return True
    else:
        return False
        
def check_directory_for_features(dir_: str)-> bool:
    """Check if features already exist in directory."""
    features_dir = join(dir_, 'features')
    if os.path.exists(features_dir):
        if len(glob.glob(join(features_dir, 'features*.pt')))>0:
            logger.warning(f"Features already exist in {features_dir}.")
            logger.warning(f"Skipping feature creation.")
        return True
    else:
        return False
    
    
def split_path(path_str: str) -> list:
    """Split path into its components."""
    directories = []
    while path_str:
        path_str, directory = os.path.split(path_str)
        # If we've reached the root directory
        if directory:
            directories.append(directory)
        elif path_str:
            break
    return directories[::-1]  # Reverse the list to get original order

def convert_epochs_to_steps(cfg: Config, key: str, num_patients: int, batch_size: int)->None:
    """Convert number of epochs to number of steps based on number of patients and batch size"""
    logger.info(f"Computing number of steps from {key}")
    num_epochs = cfg.scheduler[key]
    num_steps = int(num_patients / batch_size * num_epochs)
    logger.info(f"Number of steps for {key}: {num_steps}")
    cfg.scheduler[key.replace('_epochs', '_steps')] = num_steps
    del cfg.scheduler[key]
    
def compute_number_of_warmup_steps(cfg: Config, num_patients:int)->None:
    """Compute number of warmup steps based on number of patients and batch size"""
    batch_size = cfg.trainer_args.batch_size
    epochs_keys = [key for key in cfg.scheduler if key.endswith('_epochs')]
    for key in epochs_keys:
        convert_epochs_to_steps(cfg, key, num_patients, batch_size)

@dataclass
class Data:
    features: dict = field(default_factory=dict)
    pids: list = field(default_factory=list)
    outcomes: Optional[List] = field(default=None)
    censor_outcomes: Optional[List] = field(default=None)
    vocabulary: Optional[Dict] = field(default=None)
    mode: Optional[str] = field(default=None)
    
    def __len__(self):
        return len(self.pids)
    
    def copy(self) -> 'Data':
        """Create a copy of this Data object"""
        return Data(
            features=deepcopy(self.features),
            pids=deepcopy(self.pids),
            outcomes=deepcopy(self.outcomes) if self.outcomes is not None else None,
            censor_outcomes=deepcopy(self.censor_outcomes) if self.censor_outcomes is not None else None,
            vocabulary=deepcopy(self.vocabulary),
            mode=self.mode
        )
    @classmethod
    def load_from_directory(cls, data_dir:str, mode:str)->'Data':
        """Load data from data_dir."""
        def load_tensor(filename, required=False):
            """Helper function to load a tensor if it exists, otherwise return None"""
            filepath = join(data_dir, filename)
            if not os.path.exists(filepath):
                if required:
                    raise FileNotFoundError(f"{filename} not found in {data_dir}")
                else:
                    return None
            return torch.load(filepath)
        features = load_tensor(f'{mode}_features.pt', required=True)
        pids = load_tensor(f'{mode}_pids.pt', required=True)
        outcomes = load_tensor(f'{mode}_outcomes.pt')
        censor_outcomes = load_tensor(f'{mode}_censor_outcomes.pt')
        vocabulary = load_tensor('vocabulary.pt')
        return cls(features, pids, outcomes, censor_outcomes, vocabulary, mode=mode)
    
    def check_lengths(self):
        """Check that all features have the same length"""
        for key, values in self.features.items():
            assert len(values) == len(self.pids), f"Length of {key} does not match length of pids"
        if self.outcomes is not None:
            assert len(self.outcomes) == len(self.pids), "Length of outcomes does not match length of pids"
        if self.censor_outcomes is not None:
            assert len(self.censor_outcomes) == len(self.pids), "Length of censor outcomes does not match length of pids"

    def split(self, val_split: float)->Tuple['Data', 'Data']:
        """Split data into train and validation. Returns two Data objects"""
        train_indices, val_indices = self._get_train_val_splits(val_split)

        train_data = self.select_data_subset_by_indices(train_indices, 'train')
        val_data = self.select_data_subset_by_indices(val_indices, 'val')
        return train_data, val_data
    
    def select_data_subset_by_indices(self, indices: list, mode:str ='')->'Data':
        return Data(features={key: [values[i] for i in indices] for key, values in self.features.items()}, 
                        pids=[self.pids[i] for i in indices],
                        outcomes=[self.outcomes[i] for i in indices] if self.outcomes is not None else None,
                        censor_outcomes=[self.censor_outcomes[i] for i in indices] if self.censor_outcomes is not None else None,
                        vocabulary=self.vocabulary,
                        mode=mode)
    
    def select_data_subset_by_pids(self, pids: list, mode:str='')->'Data':
        pid2index = {pid: index for index, pid in enumerate(self.pids)}
        if not set(pids).issubset(set(self.pids)):
            difference = len(set(pids).difference(set(self.pids)))
            logger.warning("Selection pids for split {} is not a subset of the pids in the data. There are {} selection pids that are not in data pids.".format(mode, difference))
        logger.info(f"{len(pid2index)} pids in data")
        indices = [pid2index[pid] for pid in pids if pid in pid2index]
        logger.info(f"Selected {len(indices)} pids for split {mode}")
        return self.select_data_subset_by_indices(indices, mode)

    def _get_train_val_splits(self, split: float)->Tuple[list, list]:
        """Randomly split a list of items into two lists of lengths determined by split"""
        assert split < 1 and split > 0, "Split must be between 0 and 1"
        indices = list(range(len(self.pids)))
        random.seed(42)
        random.shuffle(indices)
        split_index = int(len(indices)*(1-split))
        return indices[:split_index], indices[split_index:]
