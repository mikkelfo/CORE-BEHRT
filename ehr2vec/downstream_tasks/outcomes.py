import torch
import os
import numpy as np
import pandas as pd
from datetime import datetime


class OutcomeMaker:
    def __init__(self, config: dict, features_cfg: dict):
        self.outcomes = config.outcomes
        self.features_cfg = features_cfg
        self.config = config

    def __call__(
        self, concepts_plus: pd.DataFrame, patients_info: pd.DataFrame, patient_set=None
    ):
        
        concepts_plus = self.remove_missing_timestamps(concepts_plus)
        patients_info_dict = patients_info.set_index("PID").to_dict()
    
        if patient_set is None:
            patient_set = self.load_patient_set()
        outcome_df = pd.DataFrame({"PID": patient_set})

        origin_point = datetime(**self.features_cfg.features.abspos)

        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]

            if types == "patients_info":
                timestamps = self.match_patient_info(outcome_df, patients_info_dict, matches)
            else:
                timestamps = self.match_concepts(concepts_plus, types, matches, attrs)

            timestamps = timestamps.rename(outcome)
            timestamps = self.get_relative_timestamps_in_hours(timestamps, origin_point)
            outcome_df = outcome_df.merge(timestamps, on="PID", how="left")
        outcomes = outcome_df.to_dict("list")
        # pids = outcomes.pop("PID")

        return outcomes
    
    @staticmethod
    def remove_missing_timestamps(concepts_plus):
        return concepts_plus[concepts_plus.TIMESTAMP.notna()]

    def get_relative_timestamps_in_hours(self, timestamps, origin_point):
        return (timestamps - origin_point).dt.total_seconds() / 60 / 60

    def match_patient_info(self, outcome_df, patients_info_dict, matches):
        timestamps = outcome_df.PID.map(
                    lambda pid: patients_info_dict[matches].get(pid, pd.NaT)
        )  # Get from dict [outcome] [pid]
        timestamps = pd.Series(
            timestamps.values, index=outcome_df.PID
        )  # Convert to series
        return timestamps

    def match_concepts(self, concepts_plus, types, matches, attrs):
        """It first goes through all the types and returns true for a row if the entry starts with any of the matches.
        We then ensure all the types are true for a row by using bitwise_and.reduce. E.g. CONCEPT==COVID_TEST AND VALUE==POSITIVE"""
        col_booleans = self.get_col_booleans(concepts_plus, types, matches)
        mask = np.bitwise_and.reduce(col_booleans)
        if "negation" in attrs:
            mask = ~mask
        return self.select_first_event(concepts_plus, mask)
        
    def select_first_event(self, concepts_plus, mask):
        return concepts_plus[mask].groupby("PID").TIMESTAMP.min()
    
    @staticmethod
    def get_col_booleans(concepts_plus, types, matches)->list:
        col_booleans = []
        for typ, lst in zip(types, matches):
            col_bool = concepts_plus[typ].astype(str).str.startswith(tuple(lst), False)
            col_booleans.append(col_bool)
        return col_booleans
    
    def load_patient_set(self):
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