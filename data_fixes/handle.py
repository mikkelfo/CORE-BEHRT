import pandas as pd


class Handler():
    def __call__(self, features: dict, concept_fill = '[UNK]', num_fill = -100, drop=True):
        return self.handle(features, concept_fill, num_fill, drop)

    def handle(self, features: dict, concept_fill = '[UNK]', num_fill = -100, drop=True):
        for key, patients in features.items():
            filler = concept_fill if key == 'concept' else num_fill

            # Drop or replace incorrect ages (-1 <= age <= 120)
            if key == 'age':
                patients = self.handle_incorrect_ages(patients, drop=drop)

            # Replace NaNs as 'filler'
            patients = self.handle_nans(patients, filler, drop=drop)

            features[key] = patients
        
        return features

    def handle_incorrect_ages(self, ages: list, num_fill=-100, drop=True):
        handled_patients = []
        for patient in ages:
            if drop:
                handled_patients.append([age for age in patient if -1 <= age <= 120])
            else:
                handled_patients.append([age if age >= -1 and age <= 120 else num_fill for age in patient])
        
        return handled_patients


    def handle_nans(self, patients: list, filler, drop=True):
        replaced_patients = []
        for patient in patients:
            if drop:
                replaced_patients.append([v for v in patient if not pd.isna(v)])
            else:
                replaced_patients.append([filler if pd.isna(v) else v for v in patient])

        return replaced_patients

