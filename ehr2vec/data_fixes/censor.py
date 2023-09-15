import pandas as pd
from data_fixes.exclude import Excluder


class Censorer:
    def __init__(self, n_hours: int, min_len: int = 3, vocabulary:dict=None) -> None:
        self.n_hours = n_hours
        self.excluder = Excluder(min_len=min_len, vocabulary=vocabulary)
        
    def __call__(self, features: dict, censor_outcomes: list) -> dict:
        features = self.censor(features, censor_outcomes)
        features, _, kept_indices = self.excluder(features, None)
        return features, kept_indices

    def censor(self, features: dict, censor_outcomes: list) -> dict:
        censored_features = {key: [] for key in features}
        for i, patient in enumerate(self._iter_patients(features)):
            censor_timestamp = censor_outcomes[i]
            censored_patient = self._censor(patient, censor_timestamp)

            # Append to censored features
            for key, value in censored_patient.items():
                censored_features[key].append(value)

        return censored_features

    def _censor(self, patient: dict, event_timestamp: float) -> dict:
        if not pd.isna(event_timestamp):
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