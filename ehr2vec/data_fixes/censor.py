import random
from typing import List, Union

import pandas as pd
from common.utils import iter_patients
from data_fixes.exclude import Excluder


class Censorer:
    def __init__(self, n_hours: int, min_len: int = 3, vocabulary:dict=None) -> None:
        """Censor the features based on the event timestamp.
        n_hours if positive, censor all items that occur n_hours after event."""
        self.n_hours = n_hours
        self.vocabulary = vocabulary
        self.excluder = Excluder(min_len=min_len, vocabulary=vocabulary)

    def __call__(self, features: dict, censor_outcomes: list, exclude: bool = True) -> dict:
        features = self.censor(features, censor_outcomes)
        if exclude:
            features, _, kept_indices = self.excluder(features, None)
            return features, kept_indices
        else:
            return features

    def censor(self, features: dict, censor_outcomes: list) -> dict:
        censored_features = {key: [] for key in features}
        for i, patient in enumerate(iter_patients(features)):
            censor_timestamp = censor_outcomes[i]
            censored_patient = self._censor(patient, censor_timestamp)

            # Append to censored features
            for key, value in censored_patient.items():
                censored_features[key].append(value)

        return censored_features

    def _censor(self, patient: dict, event_timestamp: float) -> dict:
        """Censor the patient's features based on the event timestamp."""
        if not pd.isna(event_timestamp):
            # Extract the attention mask and determine the number of non-masked items
            attention_mask = patient["attention_mask"]
            num_non_masked = sum(attention_mask)

            # Extract absolute positions and concepts for non-masked items
            absolute_positions = patient["abspos"][:num_non_masked]
            concepts = patient["concept"][:num_non_masked]

            # Determine if the concepts are tokenized and if they are background
            tokenized_concepts = self._identify_if_tokenized(concepts)
            background_flags = self._identify_background(concepts, tokenized_concepts)
            
            # Determine which items to censor based on the event timestamp and background flags
            censor_flags = self._generate_censor_flags(absolute_positions, background_flags, event_timestamp)
        
            for key, value in patient.items():
                patient[key] = [item for index, item in enumerate(value) if censor_flags[index]]
                
        return patient
    
    def _generate_censor_flags(self, absolute_positions: List[float], background_flags: List[bool], event_timestamp: float) -> List[bool]:
        """Generate flags indicating which items to censor."""
        return [
            position - event_timestamp - self.n_hours <= 0 or is_background
            for position, is_background in zip(absolute_positions, background_flags)
        ]

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

    def _identify_if_tokenized(self, concepts:list)->bool:
        """Identify if the features are tokenized."""
        return concepts and isinstance(concepts[0], int)


class EQ_Censorer(Censorer):
        
    def __call__(self, features: dict, censor_outcomes: list, exclude: bool = True) -> dict:
        censor_outcomes = self.get_censor_outcomes_for_negatives(censor_outcomes)
        return super().__call__(features, censor_outcomes, exclude)

    @staticmethod
    def get_censor_outcomes_for_negatives(censor_outcomes: list) -> list:
        """Use distribution of censor times to generate censor times for negative patients."""
        positive_censor_outcomes = [t for t in censor_outcomes if pd.notna(t)]
        return [t if pd.notna(t) else random.choice(positive_censor_outcomes) for t in censor_outcomes]

