import logging
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
from common.config import Config
from common.utils import Data

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
BG_GENDER_KEYS = {
    'male': ['M', 'Kvinde', 'male', 'Male', 'man', 'MAN', '1'],
    'female': ['W', 'Mand', 'F', 'female', 'woman', 'WOMAN', '0']
}
MIN_POSITIVES = {'train': 10, 'val': 5}
CHECKPOINT_FOLDER = 'checkpoints'

class Utilities():
    def process_datasets(self, datasets: Dict, func: callable, args_for_func: Dict=None)->Dict:
        """Apply a function to all datasets in a dictionary"""
        if args_for_func is None:
            args_for_func = {}
        for split, data in datasets.items():
            # Get mode-specific arguments, or an empty dictionary if they don't exist
            mode_args = args_for_func.get(split, {})
            datasets[split] = func(data, **mode_args)
        self.log_patient_nums(func.__name__, datasets)
        return datasets
    @staticmethod
    def log_patient_nums(operation:str, datasets: Dict)->None:
        logger.info(f"After applying {operation}:")
        for split, data in datasets.items():
            logger.info(f"{split}: {len(data.pids)} patients")
    @staticmethod
    def select_and_order_outcomes_for_patients(all_outcomes: Dict, pids: List, outcome: str) -> List:
        """Select outcomes for patients and order them based on the order of pids"""
        outcome_pids = all_outcomes[PID_KEY]
        outcome_group = all_outcomes[outcome]
        assert len(outcome_pids) == len(outcome_group), "Mismatch between PID_KEY length and outcome_group length"

        # Create a dictionary of positions for each PID for quick lookup
        pid_to_index = {pid: idx for idx, pid in enumerate(outcome_pids)}
        
        outcome_pids = set(outcome_pids)
        if not set(pids).issubset(outcome_pids):
            logger.warn(f"PIDs is not a subset of outcome PIDs, there is a mismatch of {len(set(pids).difference(outcome_pids))} patients") 
        
        outcomes = [outcome_group[pid_to_index[pid]] if pid in outcome_pids else None for pid in pids]
        return outcomes
    
    @staticmethod
    def check_and_adjust_max_segment(data: Data, model_cfg: Config)->None:
        """Check max segment. If it's larger or equal to the model type vocab size, change accordingly."""
        max_segment = max([max(seg) for seg in data.features['segment']])
        type_vocab_size = model_cfg.type_vocab_size
        if max_segment>=type_vocab_size:
            logger.warning(f"You've set type_vocab_size too low. Max segment {max_segment} >= type_vocab_size {type_vocab_size}\
                             We'll change it to {max_segment+1}.")
            model_cfg.type_vocab_size = max_segment+1
    @staticmethod
    def get_token_to_index_map(vocabulary:dict)->Tuple[dict]:
        """
        Creates a new mapping from vocbulary values to new integers excluding special tokens
        """
        filtered_tokens = set([v for k, v in vocabulary.items() if not k.startswith('[')])
        token2index = {token: i for i, token in enumerate(filtered_tokens)}        
        new_vocab = {k: token2index[v] for k, v in vocabulary.items() if v in token2index}
        return token2index, new_vocab
    @staticmethod
    def get_gender_token(vocabulary: dict, key: str)->int:
        """Get the token from the vocabulary corresponding to the gender provided in the config"""
        # Determine the gender category
        gender_category = None
        for category, keys in BG_GENDER_KEYS.items():
            if key in keys:
                gender_category = category
                break

        if gender_category is None:
            raise ValueError(f"Unknown gender {key}, please select one of {BG_GENDER_KEYS}")

        # Check the vocabulary for a matching token
        for possible_key in BG_GENDER_KEYS[gender_category]:
            gender_token = vocabulary.get('BG_GENDER_' + possible_key, None)
            if gender_token is not None:
                return gender_token
    
        raise ValueError(f"None of BG_GENDER_+{BG_GENDER_KEYS[gender_category]} found in vocabulary.")
    @staticmethod
    def get_background_indices(data: Data)->List[int]:
        """Get the length of the background sentence"""
        background_tokens = set([v for k, v in data.vocabulary.items() if k.startswith('BG_')])
        example_concepts = data.features['concept'][0] # Assume that all patients have the same background length
        background_indices = [i for i, concept in enumerate(example_concepts) if concept in background_tokens]
        if data.vocabulary['[SEP]'] in example_concepts:
            background_indices.append(max(background_indices)+1)
        return background_indices
    @staticmethod
    def code_starts_with(code: int, prefixes: set)->bool:
        """Check if the code starts with any of the given prefixes."""
        return any(code.startswith(prefix) for prefix in prefixes)
    @staticmethod
    def log_pos_patients_num(datasets: Dict)->None:
        for mode, data in datasets.items():
            num_positive_patiens = len([t for t in data.outcomes if not pd.isna(t)])
            if num_positive_patiens < MIN_POSITIVES[mode]:
                raise ValueError(f"Number of positive patients is less than {MIN_POSITIVES[mode]}: {num_positive_patiens}")
            logger.info(f"Positive {mode} patients: {num_positive_patiens}")
    @staticmethod
    def get_last_checkpoint_epoch(checkpoint_folder: str)->int:
        """Returns the epoch of the last checkpoint."""
        # Regular expression to match the pattern retry_XXX
        pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
        max_epoch = None
        for filename in os.listdir(checkpoint_folder):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch
        # Return the folder with the maximum retry number
        if max_epoch is None:
            raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))
        return max_epoch
