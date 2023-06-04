import pandas as pd


class Handler():
    def __init__(self):
        self.concept_fill = '[UNK]'
        self.num_fill = -100
        self.drop = True
    def __call__(self, features: dict):
        return self.handle(features)

    def handle(self, features: dict):
        handled_patients = {k: [] for k in features}
        for i in range(len(features['concept'])):
            patient = {k: v[i] for k, v in features.items()}

            patient = self.handle_incorrect_ages(patient)

            patient = self.handle_nans(patient)

            patient['segment'] = self.normalize_segments(patient['segment'])

            for key, values in patient.items():
                handled_patients[key].append(values)

        return handled_patients

    def handle_incorrect_ages(self, patient: dict):
        correct_indices = [i for i, age in enumerate(patient['age']) if -1 <= age <= 120]
        if self.drop:
            for key, values in patient.items():
                patient[key] = [values[i] for i in correct_indices]
        else:
            for key, values in patient.items():
                patient[key] = [values[i] if i in correct_indices else self.num_fill for i in range(len(values))]

        return patient

    def handle_nans(self, patient: dict):
        if self.drop:
            nan_indices = []
            for values in patient.values():
                nan_indices.extend([i for i, v in enumerate(values) if pd.isna(v)])

            for key, values in patient.items():
                patient[key] = [v for i, v in enumerate(values) if i not in set(nan_indices)]
        
        else:
            for key, values in patient.items():
                filler = self.concept_fill if key == 'concept' else self.num_fill
                patient[key] = [filler if pd.isna(v) else v for v in values]

        return patient

    @staticmethod
    def normalize_segments(segments: list) -> list:
        segment_set = sorted(set(segments))
        correct_segments = list(range(len(segment_set)))
        converter = {k: v for (k,v) in zip(segment_set, correct_segments)}

        return [converter[segment] for segment in segments]

