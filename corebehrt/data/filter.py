import torch
import random
import pandas as pd
from os.path import join
from typing import List, Tuple

from corebehrt.data.utils import Utilities
from corebehrt.data_fixes.exclude import Excluder
from corebehrt.common.utils import Data, iter_patients
from corebehrt.common.config import Config


SPECIAL_CODES = ['[', 'BG_']

class CodeTypeFilter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.SPECIAL_CODES = SPECIAL_CODES

    def filter(self, data: Data) -> Data:
        """Filter code types, e.g. keep only diagnoses. Remove patients with not sufficient data left."""
        keep_codes = self._combine_to_tuple(self.SPECIAL_CODES, self.cfg.data.code_types)
        keep_tokens = set([token for code, token in data.vocabulary.items() if code.startswith(keep_codes)])
        for patient_index, patient in enumerate(iter_patients(data.features)):
            self._filter_patient(data, patient, keep_tokens, patient_index)
        data.vocabulary = self._filter_vocabulary(data.vocabulary, keep_tokens)

        return data

    @staticmethod
    def _filter_patient(data: Data, patient: dict, keep_tokens: set, patient_index: int) -> None:
        """Filter patient in place by removing tokens that are not in keep_tokens"""
        concepts = patient['concept']
        keep_entries = {i: token for i, token in enumerate(concepts) if token in keep_tokens}
        for k, v in patient.items():
            data.features[k][patient_index] = [v[i] for i in keep_entries]

    def _filter_vocabulary(self, vocabulary: dict, keep_tokens: set) -> dict:
        """Filter vocabulary in place by removing tokens that are not in keep_tokens"""
        keep_codes = set([code for code, token in vocabulary.items() if token in keep_tokens])
        filtered_vocabulary = {code: token for code, token in vocabulary.items() if code in keep_codes}
        # Re-index vocabulary to be sequential
        filtered_vocabulary = {code: i for i, code in enumerate(filtered_vocabulary)}

        return filtered_vocabulary

    @staticmethod
    def _combine_to_tuple(*args: Tuple[List[str]]) -> tuple:
        """Return a tuple of codes to keep according to cfg.data.code_types."""
        flatten_args = sum(args, [])
        return tuple(flatten_args)
    

class PatientFilter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def filter_outcome_before_censor(self, data: Data) -> Data:
        """Filter patients with outcome before censoring and missing censoring when outcome present."""
        kept_indices = []
        for i, (outcome, censor) in enumerate(zip(data.outcomes, data.censor_outcomes)):
            if pd.isna(censor) and pd.isna(outcome):
                kept_indices.append(i)
            elif pd.isna(outcome) or outcome >= (censor + self.cfg.outcome.n_hours):
                kept_indices.append(i)
        return self.select_entries(data, kept_indices)
    
    def select_censored(self, data: Data) -> Data:
        """Select only censored patients. This is only relevant for the fine-tuning  data. 
        E.g. for pregnancy complications select only pregnant women."""
        kept_indices = [i for i, censor in enumerate(data.censor_outcomes) if pd.notna(censor)]
        return self.select_entries(data, kept_indices)

    def exclude_short_sequences(self, data: Data) -> Data:
        """Exclude patients with less than k concepts"""
        excluder = Excluder(min_len = self.cfg.data.get('min_len', 3),
                            vocabulary=data.vocabulary)
        kept_indices = excluder._exclude(data.features)
        return self.select_entries(data, kept_indices)

    def select_by_age(self, data: Data) -> Data:
        """
        We retrieve the age of each patient at censor date and check whether it's within the range.
        """
        kept_indices = []
        min_age = self.cfg.data.get('min_age', 0)
        max_age = self.cfg.data.get('max_age', 120)

        # Calculate ages at censor date for all patients
        ages_at_censor_date = Utilities.calculate_ages_at_censor_date(data)
        kept_indices = [i for i, age in enumerate(ages_at_censor_date) 
                if min_age <= age <= max_age]
        return self.select_entries(data, kept_indices)

    def select_by_gender(self, data: Data) -> Data:
        """Select only patients of a certain gender"""
        gender_token = Utilities.get_gender_token(data.vocabulary, self.cfg.data.gender)
        kept_indices = [i for i, concepts in enumerate(data.features['concept']) if gender_token in set(concepts)]
        return self.select_entries(data, kept_indices)
    
    def select_random_subset(self, data, num_patients, seed=42) -> Data:
        """Select a num_patients random patients"""
        if len(data.pids) <= num_patients:
            return data
        indices = list(range(len(data.pids)))
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:num_patients]
        return self.select_entries(data, indices)

    @staticmethod
    def select_entries(data:Data, indices:List) -> Data:
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
    
    @staticmethod
    def exclude_pids(data: Data, exclude_pids: List[str]) -> Data:
        """Exclude pids from data."""
        current_pids = data.pids
        data = data.select_data_subset_by_pids(list(set(current_pids).difference(set(exclude_pids))), mode=data.mode)
        return data

