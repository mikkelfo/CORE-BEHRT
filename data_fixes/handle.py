import pandas as pd


class Handler():
    def __call__(self, features: dict, concept_fill = '[UNK]', num_fill = -100, drop=True):
        return self.handle(features, concept_fill, num_fill, drop)

    def handle(self, features: dict, concept_fill = '[UNK]', num_fill = -100, drop=True):
        handled_patients = {k: [] for k in features}
        for i in range(len(features['concept'])):
            patient = {k: v[i] for k, v in features.items()}

            patient = self.handle_incorrect_ages(patient, num_fill=num_fill, drop=drop)

            patient = self.handle_nans(patient, concept_fill=concept_fill, num_fill=num_fill, drop=drop)

            for key, values in patient.items():
                handled_patients[key].append(values)

        return handled_patients


    def handle_incorrect_ages(self, patient: dict, num_fill=-100, drop=True):
        correct_indices = [i for i, age in enumerate(patient['age']) if -1 <= age <= 120]
        if drop:
            for key, values in patient.items():
                patient[key] = [values[i] for i in correct_indices]
        else:
            for key, values in patient.items():
                patient[key] = [values[i] if i in correct_indices else num_fill for i in range(len(values))]

        return patient

    def handle_nans(self, patient: dict, concept_fill = '[UNK]', num_fill = -100, drop=True):
        if drop:
            nan_indices = []
            for values in patient.values():
                nan_indices.extend([i for i, v in enumerate(values) if pd.isna(v)])

            for key, values in patient.items():
                patient[key] = [v for i, v in enumerate(values) if i not in set(nan_indices)]
        
        else:
            for key, values in patient.items():
                filler = concept_fill if key == 'concept' else num_fill
                patient[key] = [filler if pd.isna(v) else v for v in values]

        return patient

