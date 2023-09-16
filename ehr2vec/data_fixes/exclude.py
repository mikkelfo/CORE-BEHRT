import pandas as pd

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
        kept_indices = self._exclude(features)
        
        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]
        if outcomes:
            outcomes = [outcomes[i] for i in kept_indices]
        return features, outcomes, kept_indices

    def _exclude(self, features: dict):
        kept_indices = []
        tokenized_features = False
        if isinstance(features['concept'][0][0], int):
            tokenized_features = True
            special_tokens = set([_int for str_, _int in self.vocabulary.items() if str_.startswith('[')])
        for i, concepts in enumerate(features['concept']):
            if tokenized_features:
                unique_codes = set([code for code in concepts if not code in special_tokens])
            else:
                unique_codes = set([code for code in concepts if not code.startswith('[')])
            if len(unique_codes) >= self.min_len:
                kept_indices.append(i)
        return kept_indices

    @staticmethod
    def exclude_covid_negative(features: dict, outcomes: dict):
        kept_indices = []
        for i, result in enumerate(outcomes['COVID']):
            if pd.notna(result):
                kept_indices.append(i)

        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]

        for key, values in outcomes.items():
            outcomes[key] = [values[i] for i in kept_indices]

        return features, outcomes