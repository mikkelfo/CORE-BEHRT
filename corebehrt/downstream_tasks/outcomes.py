import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from corebehrt.data.utils import Utilities


class OutcomeMaker:
    def __init__(self, config: dict, features_cfg: dict):
        self.outcomes = config.outcomes
        self.features_cfg = features_cfg
        self.config = config

    def __call__(
        self, concepts_plus: pd.DataFrame, patients_info: pd.DataFrame, patient_set=None
    )->dict:
        
        concepts_plus = self.remove_missing_timestamps(concepts_plus)
        patients_info_dict = patients_info.set_index("PID").to_dict()
    
        if patient_set is None:
            patient_set = self.load_patient_set()
        outcome_df = pd.DataFrame({"PID": patient_set})

        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]
            if types == "patients_info":
                timestamps = self.match_patient_info(outcome_df, patients_info_dict, matches)
            else:
                timestamps = self.match_concepts(concepts_plus, types, matches, attrs)
            timestamps = timestamps.rename(outcome)
            timestamps = Utilities.get_abspos_from_origin_point(timestamps, self.features_cfg.features.abspos)
            outcome_df = outcome_df.merge(timestamps, on="PID", how="left")
        outcomes = outcome_df.to_dict("list")

        return outcomes
    
    @staticmethod
    def remove_missing_timestamps(concepts_plus: pd.DataFrame )->pd.DataFrame:
        return concepts_plus[concepts_plus.TIMESTAMP.notna()]

    def match_patient_info(self, outcome: pd.DataFrame, patients_info: dict, matches: List[List])->pd.Series:
        timestamps = outcome.PID.map(
                    lambda pid: patients_info[matches].get(pid, pd.NaT)
        )  # Get from dict [outcome] [pid]
        timestamps = pd.Series(
            timestamps.values, index=outcome.PID
        )  # Convert to series
        return timestamps

    def match_concepts(self, concepts_plus: pd.DataFrame, types: List[List], matches:List[List], attrs:Dict):
        """It first goes through all the types and returns true for a row if the entry starts with any of the matches.
        We then ensure all the types are true for a row by using bitwise_and.reduce. E.g. CONCEPT==COVID_TEST AND VALUE==POSITIVE"""
        if 'exclude' in attrs:
            concepts_plus = concepts_plus[~concepts_plus['CONCEPT'].isin(attrs['exclude'])]
        col_booleans = self.get_col_booleans(concepts_plus, types, matches, 
                                             attrs.get("match_how", 'startswith'), attrs.get("case_sensitive", True))
        mask = np.bitwise_and.reduce(col_booleans)
        if "negation" in attrs:
            mask = ~mask
        if attrs.get("use_last", False):
            return self.select_last_event(concepts_plus, mask)
        return self.select_first_event(concepts_plus, mask)
    @staticmethod
    def select_last_event(concepts_plus:pd.DataFrame, mask:pd.Series):
        return concepts_plus[mask].groupby("PID").TIMESTAMP.max()
    @staticmethod
    def select_first_event(concepts_plus:pd.DataFrame, mask:pd.Series):
        return concepts_plus[mask].groupby("PID").TIMESTAMP.min()
    
    @staticmethod
    def get_col_booleans(concepts_plus:pd.DataFrame, types:List, matches:List[List], 
                         match_how:str='startswith', case_sensitive:bool=True)->list:
        col_booleans = []
        for typ, lst in zip(types, matches):
            if match_how=='startswith':
                if case_sensitive:
                    col_bool = concepts_plus[typ].astype(str).str.startswith(tuple(lst), False)
                else:
                    match_lst = [x.lower() for x in lst]
                    col_bool = concepts_plus[typ].astype(str).str.lower().str.startswith(tuple(match_lst), False)
            elif match_how == 'contains':
                col_bool = pd.Series([False] * len(concepts_plus), index=concepts_plus.index)
                for item in lst:
                    pattern = item if case_sensitive else item.lower()
                    if case_sensitive:
                        col_bool |= concepts_plus[typ].astype(str).str.contains(pattern, na=False)
                    else:
                        col_bool |= concepts_plus[typ].astype(str).str.lower().str.contains(pattern, na=False)
            else:
                raise ValueError(f"match_how must be startswith or contains, not {match_how}")
            col_booleans.append(col_bool)
        return col_booleans
    
    def load_patient_set(self)->list:
        pids = torch.load(
                os.path.join(self.config.paths.extra_dir, "PIDs.pt")
        )  # Load PIDs
        excluder_kept_indices = torch.load(
            os.path.join(self.config.paths.extra_dir, "excluder_kept_indices.pt")
        )  # Remember excluded patients
        patient_set = [
            pids[i] for i in excluder_kept_indices
        ]  # Construct patient set
        return patient_set