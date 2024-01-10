import torch
import logging
import random
# import numpy as np
import pandas as pd
from os.path import join
from typing import List, Tuple


from data.utils import Utilities
from data_fixes.exclude import Excluder
from common.utils import Data, iter_patients

logger = logging.getLogger(__name__)  # Get the logger for this module

SPECIAL_CODES = ['[', 'BG_']

class CodeTypeFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.utils = Utilities
        self.SPECIAL_CODES = SPECIAL_CODES

    def filter(self, data: Data)->Data:
        """Filter code types, e.g. keep only diagnoses. Remove patients with not sufficient data left."""
        keep_codes = self._combine_to_tuple(self.SPECIAL_CODES, self.cfg.data.code_types)
        keep_tokens = set([token for code, token in data.vocabulary.items() if self.utils.code_starts_with(code, keep_codes)])
        logger.info(f"Keep only codes starting with: {keep_codes}")
        for patient_index, patient in enumerate(iter_patients(data.features)):
            self._filter_patient(data, patient, keep_tokens, patient_index)
        return data

    @staticmethod
    def _filter_patient(data: Data, patient: dict, keep_tokens: set, patient_index: int)->None:
        """Filter patient in place by removing tokens that are not in keep_tokens"""
        concepts = patient['concept']
        keep_entries = {i:token for i, token in enumerate(concepts) if token in keep_tokens}
        for k, v in patient.items():
            filtered_list = [v[i] for i in keep_entries]
            data.features[k][patient_index] = filtered_list

    @staticmethod
    def _combine_to_tuple(*args: Tuple[List[str]])->tuple:
        """Return a tuple of codes to keep according to cfg.data.code_types."""
        flatten_args = sum(args, [])
        return tuple(flatten_args)
    

class PatientFilter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.utils = Utilities

    def exclude_pretrain_patients(self, data: Data)->Data:
        """Exclude patients from pretraining set."""
        pretrain_pids = set()
        for mode in ['train', 'val']:
            pretrain_pids.update(set(torch.load(join(self.cfg.paths.pretrain_model_path, f'pids_{mode}.pt'))))
        kept_indices = [i for i, pid in enumerate(data.pids) if pid not in pretrain_pids]
        return self.select_entries(data, kept_indices)

    def filter_outcome_before_censor(self, data: Data)->Data:
        """Filter patients with outcome before censoring and missing censoring when outcome present."""
        kept_indices = []
        for i, (outcome, censor) in enumerate(zip(data.outcomes, data.censor_outcomes)):
            if pd.isna(censor):
                if pd.isna(outcome):
                    kept_indices.append(i)
            elif pd.isna(outcome) or outcome >= (censor + self.cfg.outcome.n_hours):
                kept_indices.append(i)
        return self.select_entries(data, kept_indices)
    
    def select_censored(self, data: Data)->Data:
        """Select only censored patients. This is only relevant for the fine-tuning  data. 
        E.g. for pregnancy complications select only pregnant women."""
        kept_indices = [i for i, censor in enumerate(data.censor_outcomes) if pd.notna(censor)]
        return self.select_entries(data, kept_indices)

    def exclude_short_sequences(self, data: Data)->Data:
        """Exclude patients with less than k concepts"""
        excluder = Excluder(min_len = self.cfg.data.get('min_len', 3),
                            vocabulary=data.vocabulary)
        kept_indices = excluder._exclude(data.features)
        return self.select_entries(data, kept_indices)
    
    def select_by_age(self, data: Data)->Data:
        """
        Assuming that age is given in days. 
        We retrieve the last age of each patient and check whether it's within the range.
        """
        kept_indices = []
        min_age = self.cfg.data.get('min_age', 0)
        max_age = self.cfg.data.get('max_age', 120)
        kept_indices = [i for i, ages in enumerate(data.features['age']) 
                if min_age <= ages[-1] <= max_age]
        return self.select_entries(data, kept_indices)
    
    def select_by_gender(self, data: Data)->Data:
        """Select only patients of a certain gender"""
        gender_token = self.utils.get_gender_token(data.vocabulary, self.cfg.data.gender)
        kept_indices = [i for i, concepts in enumerate(data.features['concept']) if gender_token in set(concepts)]
        return self.select_entries(data, kept_indices)
    
    def select_random_subset(self, data, num_patients, seed=42)->Data:
        """Select a num_patients random patients"""
        if len(data.pids) <= num_patients:
            return data
        random.seed(seed)
        indices = list(range(len(data.pids)))
        random.shuffle(indices)
        indices = indices[:num_patients]
        return self.select_entries(data, indices)

    @staticmethod
    def select_entries(data:Data, indices:List)->Data:
        """
        Select entries based on indices. 
        Optionally for outcomes and censor outcomes, if present returns dict of results.
        """
        indices = set(indices)
        data.features = {k: [v[i] for i in indices] for k, v in data.features.items()}
        data.pids = [data.pids[i] for i in indices]
        if data.outcomes is not None:
            data.outcomes = [data.outcomes[i] for i in indices]
        if data.censor_outcomes is not None:
            data.censor_outcomes = [data.censor_outcomes[i] for i in indices]
        return data

