import torch
import os
import numpy as np
import pandas as pd
from datetime import datetime


class OutcomeMaker:
    def __init__(self, config: dict):
        self.outcomes = config.outcomes
        self.config = config

    def __call__(
        self, concepts_plus: pd.DataFrame, patients_info: pd.DataFrame, patient_set=None
    ):
        # Remove nan TIMESTAMPs and convert cols to str
        concepts_plus = concepts_plus[concepts_plus.TIMESTAMP.notna()]

        # Convert patients_info to dict
        patients_info_dict = patients_info.set_index("PID").to_dict()

        # Init outcome dataframe
        if patient_set is None:
            pids = torch.load(
                os.path.join(self.config.paths.extra_dir, "PIDs.pt")
            )  # Load PIDs
            excluder_kept_indices = torch.load(
                os.path.join(self.config.paths.extra_dir, "excluder_kept_indices.pt")
            )  # Remember excluded patients
            patient_set = [
                pids[i] for i in excluder_kept_indices
            ]  # Construct patient set
        outcome_df = pd.DataFrame({"PID": patient_set})

        # Get origin point
        origin_point = datetime(**self.config.features.abspos)

        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]

            # If patients_info, get from dict (patients_info_dict)
            if types == "patients_info":
                timestamps = outcome_df.PID.map(
                    lambda pid: patients_info_dict[matches].get(pid, pd.NaT)
                )  # Get from dict [outcome] [pid]
                timestamps = pd.Series(
                    timestamps.values, index=outcome_df.PID
                )  # Convert to series

            # Default, get from dataframe (concepts_plus)
            else:
                # Get a boolean matrix (row x col) of whether any concept matches
                col_booleans = [
                    concepts_plus[typ].astype(str).str.startswith(tuple(lst), False)
                    for typ, lst in zip(types, matches)
                ]
                # Get a boolean vector (row) of whether all concepts match (a & b & ...)
                mask = np.bitwise_and.reduce(col_booleans)

                # Support negation
                if "negation" in attrs:
                    timestamps = concepts_plus[~mask].groupby("PID").TIMESTAMP.min()
                else:
                    timestamps = concepts_plus[mask].groupby("PID").TIMESTAMP.min()

            # Rename and convert to abspos
            timestamps = timestamps.rename(outcome)
            timestamps = (origin_point - timestamps).dt.total_seconds() / 60 / 60

            outcome_df = outcome_df.merge(timestamps, on="PID", how="left")

        # Format to dict
        outcomes = outcome_df.to_dict("list")
        del outcomes["PID"]

        return outcomes