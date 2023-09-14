import torch
from src.data_fixes.handle import Handler


class Truncator:
    def __call__(self, features: dict, max_len: int = 512, sep_token: int = 2):
        return self.truncate(features, max_len, sep_token)

    def truncate(self, features: dict, max_len: int = 512, sep_token: int = 2):
        truncated_features = {key: [] for key in features}
        for patient in self._iter_patients(features):
            patient = self._truncate_patient(patient, max_len, sep_token)
            patient["segment"] = Handler.normalize_segments(patient["segment"])

            for key, value in patient.items():
                truncated_features[key].append(value)

        return truncated_features

    @staticmethod
    def _truncate_patient(patient: dict, max_len: int = 512, sep_token: int = 2):
        # Do not truncate if patient is shorter than max_len
        if len(patient["concept"]) <= max_len:
            return patient

        # Get index of first [SEP] token (when background sentence ends) and djust for 0-indexing
        background_length = patient["concept"].index(sep_token) + 1
        truncation_length = max_len - background_length

        # Do not start seq with [SEP] token (SEP token is included in background sentence)
        if patient["concept"][-truncation_length] == sep_token:
            truncation_length -= 1

        return {
            key: value[:background_length] + value[-truncation_length:]
            for key, value in patient.items()
        }

    @staticmethod
    def _iter_patients(features: dict):
        for i in range(len(features["concept"])):
            yield {key: values[i] for key, values in features.items()}
