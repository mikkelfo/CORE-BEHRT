import torch
import pandas as pd
from src.data_fixes.exclude import Excluder


class Censor:
    def __init__(self, n_hours: int) -> None:
        self.n_hours = n_hours

    def __call__(self, features: dict, censor_outcomes: list) -> dict:
        return self.censor(features, censor_outcomes)

    def censor(self, features: dict, censor_outcomes: list) -> dict:
        censored_features = {key: [] for key in features}
        for i, patient in enumerate(self._iter_patients(features)):
            censor_timestamp = censor_outcomes[i]
            censored_patient = self._censor(patient, censor_timestamp)

            # Append to censored features
            for key, value in censored_patient.items():
                censored_features[key].append(value)

        # Re-exclude short sequences after truncation
        censored_features = Excluder.exclude_short_sequences(
            censored_features, name="censor_kept_indices"
        )

        return censored_features

    def _censor(self, patient: dict, event_timestamp: float) -> dict:
        if pd.isna(event_timestamp):
            return patient
        else:
            # Only required when padding
            mask = patient["attention_mask"]
            N_nomask = sum(mask)
            pos = patient["abspos"][:N_nomask]

            # censor the last n_hours
            dont_censor = [p - event_timestamp - self.n_hours <= 0 for p in pos]

            for key, value in patient.items():
                patient[key] = [v for i, v in enumerate(value) if dont_censor[i]]

        return patient

    @staticmethod
    def _iter_patients(features: dict) -> dict:
        for i in range(len(features["concept"])):
            yield {key: values[i] for key, values in features.items()}
