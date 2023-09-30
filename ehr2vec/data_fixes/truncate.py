

class Truncator:
    def __init__(self, max_len: int, vocabulary: dict) -> None:
        self.max_len = max_len
        self.vocabulary = vocabulary
        self.sep_token = self.vocabulary.get("[SEP]", False)
        self.cls_token = self.vocabulary.get("[CLS]", False)
    
    def __call__(self, features: dict)-> dict:
        return self.truncate(features)

    def truncate(self, features: dict):
        background_length = self._get_background_length(features)
        for index, patient in enumerate(self._iter_patients(features)):
            truncated_patient = self._truncate_patient(patient, background_length)
            for key, value in truncated_patient.items():
                features[key][index] = value
        return features

    def _truncate_patient(self, patient: dict, background_length: int)-> dict:
        """Truncate patient to max_len, keeping background if present and CLS if present."""
        # Do not truncate if patient is shorter than max_len
        if len(patient["concept"]) <= self.max_len:
            return patient
        truncation_length = self.max_len - background_length
        # Do not start seq with [SEP] token (SEP token is included in background sentence)
        if patient["concept"][-truncation_length] == self.sep_token:
            truncation_length -= 1
        if self.cls_token:
            truncation_length -= 1
            cls_token_int = 1
        else:
            cls_token_int = 0
        return {
            key: value[:cls_token_int] +\
                  value[cls_token_int:background_length+cls_token_int] +\
                      value[-truncation_length:]
            for key, value in patient.items()
        }

    def _get_background_length(self, features: dict):
        """Get the length of the background sentence, first SEP token included."""
        background_tokens = set([v for k, v in self.vocabulary.items() if k.startswith('BG_')])
        example_concepts = features['concept'][0] # Assume that all patients have the same background length
        background_length = len(set(example_concepts) & background_tokens)
        return background_length + int((background_length > 0) and self.sep_token)

    @staticmethod
    def _iter_patients(features: dict):
        for i in range(len(features["concept"])):
            yield {key: values[i] for key, values in features.items()}