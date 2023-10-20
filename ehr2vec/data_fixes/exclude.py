import pandas as pd
from typing import Tuple

class Excluder():
    def __init__(self, min_len: int, min_count: int=0, vocabulary:dict=None):
        self.min_len = min_len
        self.vocabulary = vocabulary

    def __call__(self, features: dict, outcomes: dict=None,) -> pd.DataFrame:
        # Exclude patients with few concepts
        features, outcomes, kept_indices = self.exclude_short_sequences(features, outcomes)
        return features, outcomes, kept_indices

    # Currently unused
    def exclude_rare_concepts(self, features: dict) -> pd.DataFrame:
        """Excludes concepts that occur less than min_count times."""
        raise DeprecationWarning # This function is not used
        unique_codes = {}
        for patient in features['concept']:
            for code in patient:
                unique_codes[code] = unique_codes.get(code, 0) + 1
        exclude = {code for code, count in unique_codes.items() if count < self.min_count}

        for i, patient in enumerate(features['concept']):
            kept_indices = [idx for idx, code in enumerate(patient) if not code in exclude]
            for key, values in features.items():
                features[key][i] = [values[i][j] for j in kept_indices]

        return features
    
    def exclude_short_sequences(self, features: dict, outcomes: dict=None) -> pd.DataFrame:
        """Excludes patients with less than min_len concepts."""
        kept_indices = self._exclude(features)
        
        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]
        if outcomes:
            outcomes = [outcomes[i] for i in kept_indices]
        return features, outcomes, kept_indices

    def _exclude(self, features: dict)->list:
        kept_indices = []
        tokenized_features = self._is_tokenized(features['concept'])
        
        if tokenized_features:
            special_tokens = set([_int for str_, _int in self.vocabulary.items() if str_.startswith('[')])
            is_special = lambda x: x in special_tokens
        else:
            is_special = lambda x: x.startswith('[')
        
        for i, concepts in enumerate(features['concept']):
            unique_codes = set([code for code in concepts if not is_special(code)])
            if len(unique_codes) >= self.min_len:
                kept_indices.append(i)

        return kept_indices
    
    def _is_tokenized(self, concepts_list)->bool:
        """
        Determines if a list of concepts is tokenized or not.
        Returns True if tokenized, otherwise False.
        """
        for concepts in concepts_list:
            if concepts:  # Check if list is not empty
                return isinstance(concepts[0], int)
        return False
    
    @staticmethod
    def exclude_covid_negative(features: dict, outcomes: dict)->Tuple[dict, list]:
        kept_indices = []
        for i, result in enumerate(outcomes['COVID']):
            if pd.notna(result):
                kept_indices.append(i)

        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]

        for key, values in outcomes.items():
            outcomes[key] = [values[i] for i in kept_indices]

        return features, outcomes
