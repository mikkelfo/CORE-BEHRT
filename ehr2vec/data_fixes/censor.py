import pandas as pd
from data_fixes.exclude import Excluder
from typing import List, Union

class Censorer:
    def __init__(self, n_hours: int, min_len: int = 3, vocabulary:dict=None) -> None:
        self.n_hours = n_hours
        self.vocabulary = vocabulary
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

            tokenized = self._identify_if_tokenized(patient["concept"][:N_nomask])
            is_background = self._identify_background(patient["concept"][:N_nomask], tokenized)
            # censor the last n_hours
            dont_censor = [p - event_timestamp - self.n_hours <= 0 or background \
                           for p, background in zip(pos, is_background)]
            for key, value in patient.items():
                patient[key] = [v for i, v in enumerate(value) if dont_censor[i]]
        return patient
    
    def _identify_background(self, concepts: List[Union[int, str]], tokenized: bool) -> List[bool]:
        """
        Identify background items in the patient's concepts.
        Return a list of booleans of the same length as concepts indicating if each item is background.
        """
        if tokenized:
            bg_values = set([v for k, v in self.vocabulary.items() if k.startswith('BG_')])
            return [concept in bg_values for concept in concepts]
        else:
            return [concept.startswith('BG_') for concept in concepts]

    def _identify_if_tokenized(self, concepts):
        """Identify if the features are tokenized."""
        tokenized_features = False
        if concepts:
            if isinstance(concepts[0], int):
                tokenized_features = True
        return tokenized_features

    @staticmethod
    def _iter_patients(features: dict) -> dict:
        for i in range(len(features["concept"])):
            yield {key: values[i] for key, values in features.items()}