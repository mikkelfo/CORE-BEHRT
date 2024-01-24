import pandas as pd
from typing import Tuple


class Excluder:
    def __init__(self, min_len: int, vocabulary: dict=None):
        self.min_len = min_len
        self.vocabulary = vocabulary

    def __call__(self, features: dict, outcomes: dict=None,) -> pd.DataFrame:
        # Exclude patients with few concepts
        features, outcomes, kept_indices = self.exclude_short_sequences(features, outcomes)
        return features, outcomes, kept_indices
    
    def exclude_short_sequences(self, features: dict, outcomes: list = None) -> Tuple[dict, list, list]:
        """Excludes patients with less than min_len concepts."""
        kept_indices = self._exclude(features)
        
        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]
        if outcomes:
            outcomes = [outcomes[i] for i in kept_indices]
        return features, outcomes, kept_indices

    def _exclude(self, features: dict) -> list:
        kept_indices = []
        tokenized_features = self._is_tokenized(features['concept'])
        
        if tokenized_features:
            special_tokens = set([idx for key, idx in self.vocabulary.items() if key.startswith(('[', 'BG_'))])
            is_special = lambda x: x in special_tokens
        else:
            is_special = lambda x: x.startswith(('[', 'BG_'))
        
        for i, concepts in enumerate(features['concept']):
            codes = [code for code in concepts if not is_special(code)]
            if len(codes) >= self.min_len:
                kept_indices.append(i)

        return kept_indices
    
    @staticmethod
    def _is_tokenized(concepts_list: list) -> bool:
        """
        Determines if a list of concepts is tokenized or not.
        Returns True if tokenized, otherwise False.
        """
        for concepts in concepts_list:
            if concepts:  # Check if list is not empty
                return isinstance(concepts[0], int)
        return False
    

