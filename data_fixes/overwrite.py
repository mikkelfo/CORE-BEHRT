import pandas as pd


class Overwriter():
    def __call__(self, features: dict, concept_fill = '[UNK]', num_fill = -100):
        return self.overwrite(features, concept_fill, num_fill)

    def overwrite(self, features: dict, concept_fill = '[UNK]', num_fill = -100):
        for key, patients in features.items():
            filler = concept_fill if key == 'concept' else num_fill

            # Drop or replace incorrect ages (-1 <= age <= 120)
            if key == 'age':
                patients = self.handle_incorrect_ages(patients)

            # Replace NaNs as 'filler'
            patients = self.replace_nans(patients, filler)

            features[key] = patients
        
        return features

    def handle_incorrect_ages(self, ages: list, drop=True, num_fill=-100):
        handled_patients = []
        for patient in ages:
            if drop:
                handled_patients.append([age for age in patient if -1 <= age <= 120])
            else:
                handled_patients.append([age if age >= -1 and age <= 120 else num_fill for age in patient])
        
        return handled_patients


    def replace_nans(self, patients: list, filler):
        replaced_patients = []
        for patient in patients:
            replaced_patients.append([filler if pd.isna(v) else v for v in patient])

        return replaced_patients

