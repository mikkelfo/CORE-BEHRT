import torch
import os
import numpy as np
import pandas as pd


class OutcomeMaker:
    def __init__(self, config: dict):
        self.outcomes = config

    def __call__(self, concepts_plus: pd.DataFrame, patients_info: pd.DataFrame, patient_set: list=None):
        # Remove nan TIMESTAMPs and convert cols to str
        concepts_plus = concepts_plus[concepts_plus.TIMESTAMP.notna()]

        # Convert patients_info to dict
        patients_info_dict = patients_info.set_index("PID").to_dict()

        # Init outcome dataframe
        outcome_df = pd.DataFrame({"PID": patient_set})

        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]

            if types == "patients_info":
                timestamps = outcome_df.PID.map(
                    lambda pid: patients_info_dict[matches].get(pid, pd.NaT)
                )  # Get from dict [outcome] [pid]
                timestamps = pd.Series(
                    timestamps.values, index=outcome_df.PID
                )  # Convert to series
            else:
                col_booleans = [
                    concepts_plus[typ].astype(str).str.startswith(tuple(lst), False)
                    for typ, lst in zip(types, matches)
                ]
                mask = np.bitwise_and.reduce(col_booleans)

                if "negation" in attrs:
                    timestamps = concepts_plus[~mask].groupby("PID").TIMESTAMP.min()
                else:
                    timestamps = concepts_plus[mask].groupby("PID").TIMESTAMP.min()

            timestamps = timestamps.rename(outcome)

            outcome_df = outcome_df.merge(timestamps, on="PID", how="left")

        outcomes = outcome_df.to_dict("list")
        pids = outcomes.pop["PID"]

        return outcomes, pids