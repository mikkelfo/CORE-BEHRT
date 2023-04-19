import pandas as pd

class Excluder():
    def __call__(self, features: dict, outcomes: dict, k: int = 2) -> pd.DataFrame:
        return self.exclude_by_k(features, outcomes, k=k)
    
    @staticmethod
    def exclude_by_k(features: dict, outcomes: dict, k: int = 2) -> pd.DataFrame:
        kept_indices = []
        for i, concepts in enumerate(features['concept']):
            unique_codes = set([code for code in concepts if not code.startswith('[')])
            if len(unique_codes) >= k:
                kept_indices.append(i)

        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]

        for key, values in outcomes.items():
            outcomes[key] = [values[i] for i in kept_indices]

        return features, outcomes