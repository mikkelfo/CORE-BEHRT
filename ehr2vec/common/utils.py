import glob
import logging
import os
from os.path import join

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