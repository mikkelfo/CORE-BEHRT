import pandas as pd
from corebehrt.common.utils import iter_patients

class Handler:
    def __init__(self, min_age: int = -1, max_age: int = 120):
        self.min_age = min_age
        self.max_age = max_age

    def __call__(self, features: dict) -> dict:
        return self.handle(features)

    def handle(self, features: dict) -> dict:
        """Handle the features, including: incorrect ages, nans, and segments."""
        handled_patients = {k: [] for k in features}
        for patient in iter_patients(features):
            patient = self.handle_incorrect_ages(patient, min_age=self.min_age, max_age=self.max_age)

            patient = self.handle_nans(patient)
            if 'segment' in patient:
                patient['segment'] = self.normalize_segments(patient['segment'])

            for key, values in patient.items():
                handled_patients[key].append(values)

        return handled_patients

    @staticmethod
    def handle_incorrect_ages(patient: dict, min_age: int = -1, max_age: int = 120) -> dict:
        correct_indices = set([i for i, age in enumerate(patient['age']) if min_age <= age <= max_age])
        
        for key, values in patient.items():
            patient[key] = [values[i] for i in correct_indices]

        return patient

    @staticmethod
    def handle_nans(patient: dict) -> dict:
        nan_indices = []
        for values in patient.values():
            nan_indices.extend([i for i, v in enumerate(values) if pd.isna(v)])

        nan_indices = set(nan_indices)
        for key, values in patient.items():
            patient[key] = [v for i, v in enumerate(values) if i not in nan_indices]

        return patient

    @staticmethod
    def normalize_segments(segments: list) -> list:
        segment_set = sorted(set(segments))
        correct_segments = list(range(len(segment_set)))
        converter = {k: v for (k,v) in zip(segment_set, correct_segments)}

        return [converter[segment] for segment in segments]

