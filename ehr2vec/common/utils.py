import glob
import logging
import os
from os.path import join
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)  # Get the logger for this module

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