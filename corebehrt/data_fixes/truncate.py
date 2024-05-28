from corebehrt.common.utils import iter_patients

class Truncator:
    def __init__(self, max_len: int, vocabulary: dict) -> None:
        self.max_len = max_len
        self.vocabulary = vocabulary
        self.sep_token = self.vocabulary.get("[SEP]")
    
    def __call__(self, features: dict) -> dict:
        return self.truncate(features)

    def truncate(self, features: dict) -> dict:
        background_length = self._get_background_length(features)
        for index, patient in enumerate(iter_patients(features)):
            truncated_patient = self._truncate_patient(patient, background_length)
            for key, value in truncated_patient.items():
                features[key][index] = value
        return features

    def _truncate_patient(self, patient: dict, background_length: int) -> dict:
        """Truncate patient to max_len, keeping background if present and CLS if present."""
        # Do not truncate if patient is shorter than max_len
        if len(patient["concept"]) <= self.max_len:
            return patient

        truncation_length = self.max_len - background_length

        # Do not start seq with [SEP] token (SEP token is included in background sentence)
        if patient["concept"][-truncation_length] == self.sep_token:
            truncation_length -= 1

        return {
            key: value[:background_length] + value[-truncation_length:]
            for key, value in patient.items()
        }

    def _get_background_length(self, features: dict)-> int:
        """Get the length of the background sentence, first SEP token included."""
        background_tokens = set([v for k, v in self.vocabulary.items() if k.startswith('BG_')])
        example_concepts = features['concept'][0] # Assume that all patients have the same background length
        background_length = len(set(example_concepts) & background_tokens)

        return background_length + 2 # +2 for [CLS] and [SEP] tokens

