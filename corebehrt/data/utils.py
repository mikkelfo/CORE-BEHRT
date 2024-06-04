import os
import re
import numpy as np
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union

from corebehrt.common.config import Config
from corebehrt.common.utils import Data

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
BG_GENDER_KEYS = {
    'male': ['M', 'Mand',  'male', 'Male', 'man', 'MAN', '1'],
    'female': ['W', 'Kvinde', 'F', 'female', 'Female', 'woman', 'WOMAN', '0']
}
MIN_POSITIVES = {'finetune': 1, None: 1}
CHECKPOINT_FOLDER = 'checkpoints'
ORIGIN_POINT = {'year': 2020, 'month': 1, 'day': 26, 'hour': 0, 'minute': 0, 'second': 0}


class Utilities:
    @classmethod
    def process_data(cls, data: Data, func: callable, args_for_func: Dict={})->Dict:
        """Apply a function to all datasets in a dictionary"""
        data = func(data, **args_for_func)

        return data

    @staticmethod
    def select_and_order_outcomes_for_patients(all_outcomes: Dict, pids: List, outcome: Union[str, dict]) -> List:
        """Select outcomes for patients and order them based on the order of pids"""
        outcome_pids = all_outcomes[PID_KEY]
        if isinstance(outcome, str):
            outcome_group = all_outcomes[outcome]
        elif isinstance(outcome, dict): # For temporal censoring (censoring equally at a date)
            outcome_datetime = datetime(**outcome)
            logger.warning(f"Using {ORIGIN_POINT} as origin point. Check whether it is the same as used for feature creation.")
            outcome_abspos = Utilities.get_abspos_from_origin_point([outcome_datetime], ORIGIN_POINT)
            outcome_group = outcome_abspos * len(outcome_pids)
        else:
            raise ValueError(f"Unknown outcome type {type(outcome)}")
        assert len(outcome_pids) == len(outcome_group), "Mismatch between PID_KEY length and outcome_group length"

        # Create a dictionary of positions for each PID for quick lookup
        pid_to_index = {pid: idx for idx, pid in enumerate(outcome_pids)}
        
        outcome_pids = set(outcome_pids)
        if not set(pids).issubset(outcome_pids):
            logger.warn(f"PIDs is not a subset of outcome PIDs, there is a mismatch of {len(set(pids).difference(outcome_pids))} patients") 
        
        outcomes = [outcome_group[pid_to_index[pid]] if pid in outcome_pids else None for pid in pids]
        return outcomes

    @staticmethod
    def get_abspos_from_origin_point(timestamps: Union[pd.Series, List[datetime]], 
                                     origin_point: Dict[str, int])->Union[pd.Series, List[float]]:
        """Get the absolute position in hours from the origin point"""
        origin_point = datetime(**origin_point)
        if isinstance(timestamps, pd.Series):
            return (timestamps - origin_point).dt.total_seconds() / 60 / 60
        elif isinstance(timestamps, list):
            return [(timestamp - origin_point).total_seconds() / 60 / 60 for timestamp in timestamps]

    @staticmethod
    def check_and_adjust_max_segment(data: Data, model_cfg: Config)->None:
        """Check max segment. If it's larger or equal to the model type vocab size, change accordingly."""
        max_segment = max([max(seg) for seg in data.features['segment']])
        type_vocab_size = model_cfg.type_vocab_size

        if max_segment >= type_vocab_size:
            logger.warning(f"You've set type_vocab_size too low. Max segment {max_segment} >= type_vocab_size {type_vocab_size}\
                             We'll change it to {max_segment+1}.")
            model_cfg.type_vocab_size = max_segment+1

    @staticmethod
    def get_gender_token(vocabulary: dict, gender_key: str)->int:
        """Get the token from the vocabulary corresponding to the gender provided in the config"""
        # Determine the gender category
        gender_category = None
        for category, keys in BG_GENDER_KEYS.items():
            if gender_key in keys:
                gender_category = category
                break
        else:   # If gender_key is not found in any category
            raise ValueError(f"Unknown gender {gender_key}, please select one of {BG_GENDER_KEYS}")

        # Check the vocabulary for a matching token
        for possible_key in BG_GENDER_KEYS[gender_category]:
            gender_token = vocabulary.get('BG_GENDER_' + possible_key, None)
            if gender_token is not None:
                return gender_token
    
        raise ValueError(f"None of BG_GENDER_+{BG_GENDER_KEYS[gender_category]} found in vocabulary.")

    @staticmethod
    def get_last_checkpoint_epoch(checkpoint_folder: str)->int:
        """Returns the epoch of the last checkpoint."""
        # Regular expression to match the pattern retry_XXX
        pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
        epochs = [int(pattern.match(filename)) for filename in os.listdir(checkpoint_folder)]
        if not epochs:
            raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))
        return max(epochs)

    @staticmethod
    def select_and_reorder_feats_and_pids(feats: Dict[str, List], pids: List[str], select_pids: List[str])->Tuple[Dict[str, List], List[str]]:
        """Reorders pids and feats to match the order of select_pids"""
        if not set(select_pids).issubset(set(pids)):
            raise ValueError(f"There are {len(set(select_pids).difference(set(pids)))} select pids not present in pids") 
        pid2idx = {pid: index for index, pid in enumerate(pids)}
        indices_to_keep = [pid2idx[pid] for pid in select_pids] # order is important, so keep select_pids as list
        selected_feats = {}
        for key, value in feats.items():
            selected_feats[key] = [value[idx] for idx in indices_to_keep]
        return selected_feats, select_pids

    @staticmethod
    def calculate_ages_at_censor_date(data: Data) -> List[int]:
        """
        Calculates the age of patients at their respective censor dates.
        """
        ages_at_censor_date = []
        
        for abspos, age, censor_date in zip(data.features['abspos'], data.features['age'], data.censor_outcomes):
            if censor_date is None:
                ages_at_censor_date.append(age[-1]) # if no censoring, we take the last age
                continue
            # Calculate age differences and find the closest abspos index to the censor date
            time_differences_h = np.array([censor_date - ap for ap in abspos])
            # compute closest index (with regards to abspos) on the left to censor date
            closest_abspos_index = np.argmin(
                np.where(time_differences_h < 0, np.inf, time_differences_h)) 
            age_at_censor = age[closest_abspos_index] + time_differences_h[closest_abspos_index] / 24 / 365.25
            ages_at_censor_date.append(age_at_censor)
        return ages_at_censor_date

