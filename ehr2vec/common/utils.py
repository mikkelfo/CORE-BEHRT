import glob
import logging
import os
import random
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, List, Optional, Tuple

from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)  # Get the logger for this module

def iter_patients(features: dict) -> dict:
    """Iterate over patients in a features dict."""
    for i in range(len(features["concept"])):
        yield {key: values[i] for key, values in features.items()}

def check_patient_counts(concepts, patients_info, logger):
    if concepts.PID.nunique() != patients_info.PID.nunique():
            logger.warning(f"patients info contains {patients_info.PID.nunique()} patients != \
                        {concepts.PID.nunique()} unique patients in concepts")

def check_existing_splits(data_dir)-> bool:
        if os.path.exists(join(data_dir, 'train_pids.pt')) and\
            os.path.exists(join(data_dir, 'val_pids.pt')) and\
            os.path.exists(join(data_dir, 'test_pids.pt')) and\
            os.path.exists(join(data_dir, 'train_file_ids.pt')) and\
            os.path.exists(join(data_dir, 'val_file_ids.pt')) and\
            os.path.exists(join(data_dir, 'test_file_ids.pt')):
            return True
        else:
            return False
        
def check_directory_for_features(dir_):
    features_dir = join(dir_, 'features')
    if os.path.exists(features_dir):
        if len(glob.glob(join(features_dir, 'features*.pt')))>0:
            logger.warning(f"Features already exist in {features_dir}.")
            logger.warning(f"Skipping feature creation.")
        return True
    else:
        return False
    
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
    
    def select_data_subset_by_indices(self, indices: list, mode: str)->'Data':
        return Data(features={key: [values[i] for i in indices] for key, values in self.features.items()}, 
                        pids=[self.pids[i] for i in indices],
                        outcomes=[self.outcomes[i] for i in indices],
                        censor_outcomes=[self.censor_outcomes[i] for i in indices] if self.censor_outcomes is not None else None,
                        vocabulary=self.vocabulary,
                        mode=mode)



    def _get_train_val_splits(self, split: float)->Tuple[list, list]:
        """Randomly split a list of items into two lists of lengths determined by split"""
        assert split < 1 and split > 0, "Split must be between 0 and 1"
        indices = list(range(len(self.pids)))
        random.shuffle(indices)
        split_index = int(len(indices)*(1-split))
        return indices[:split_index], indices[split_index:]

class ConcatIterableDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.pids = [pid for dataset in datasets for pid in dataset.pids]
        self.file_ids = [file_id for dataset in datasets for file_id in dataset.file_ids]
    def __iter__(self):
        for dataset in self.datasets:
            yield from dataset
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])