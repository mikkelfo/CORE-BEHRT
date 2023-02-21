import torch
from transformers import BatchEncoding


class EHRTokenizer():
    def __init__(self, config, vocabulary=None):
        self.config = config
        if vocabulary is None:
            self.new_vocab = True
            self.vocabulary = {
                '[PAD]': 0,
                '[CLS]': 1, 
                '[SEP]': 2,
                '[UNK]': 3,
                '[MASK]': 4,
            }
        else:
            self.new_vocab = False
            self.vocabulary = vocabulary

    def __call__(self, features: dict(str, list(list)), padding=True, truncation=768):
        return self.batch_encode(features, padding, truncation)

    def batch_encode(self, features: dict(str, list(list)), padding=True, truncation=768):
        data = {key: [] for key in features}
        data['attention_mask'] = []

        for patient in self._patient_iterator(features):
            patient = self.insert_special_tokens(patient)                   # Insert SEP and CLS tokens

            if truncation and len(patient['concept']) > truncation:
                patient = self.truncate(patient, max_len=truncation)        # Truncate patient to max_len
            
            # Created after truncation for efficiency
            patient['attention_mask'] = [1] * len(patient['concept'])       # Initialize attention mask

            patient['concept'] = self.encode(patient['concept'])            # Encode concepts

            for key, value in patient.items():
                data[key].append(value)

        if padding:
            longest_seq = max([len(s) for s in data['concept']])            # Find longest sequence
            max_len = min(longest_seq, truncation)                          # Find min of longest sequence and truncation length
            data = self.pad(data, max_len=max_len)                          # Pad sequences to max_len

        return BatchEncoding(data, tensor_type="pt")

    def encode(self, concepts: list):
        if self.new_vocab:
            for concept in concepts:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

        return [self.vocabulary[concept] for concept in concepts]

    def truncate(self, patient: dict(str, list), max_len: int):
        # Find length of background sentence (including CLS token) - segment 0      # TODO: What is only concepts?
        background_length = len([x for x in patient.get('segment', []) if x == 0]) 
        truncation_length = max_len - background_length
        
        # Do not start seq with SEP token
        if patient['concept'][-truncation_length] == '[SEP]':
            truncation_length += 1

        for key, value in patient.items():
            patient[key] = value[:background_length] + value[-truncation_length:]    # Keep CLS, background sentence and newest information

        return patient

    def pad(self, features: dict(str, list(list)), max_len: int):
        padded_data = {key: [] for key in features}
        for patient in self._patient_iterator(features):
            difference = max_len - len(patient['concept'])

            for key, values in patient.items():
                token = '[PAD]' if key == 'concept' else 0
                padded_data[key].append(values + [token] * difference)

        return padded_data

    def insert_special_tokens(self, patient: dict(str, list)):
        if self.config.sep_tokens:
            if 'segment' not in patient:
                raise Exception('Cannot insert SEP tokens without segment information')
            patient = self.insert_sep_tokens(patient)

        if self.config.cls_token:
            patient = self.insert_cls_token(patient)
        
        return patient

    def insert_sep_tokens(self, patient: dict(str, list)):
        def _insert_sep_tokens(seq: list):
            new_seq = []
            for i in range(len(seq)):
                new_seq.append(seq[i])

                # Insert SEP token if segment changes
                if padded_segment[i] != padded_segment[i+1]:
                    new_seq.append(token)
            return new_seq
        padded_segment = patient['segment'] + [None]                # Add None to last entry to avoid index out of range

        for key, values in patient.items():
            token = '[SEP]' if key == 'concept' else 0
            patient[key] = [_insert_sep_tokens(val) for val in values]

        return patient

    def insert_cls_token(self, patient: dict(str, list)):
        for key, value in patient.items():
            token = '[CLS]' if key == 'concept' else 0          # Determine token value (CLS for concepts, 0 for rest)
            patient[key] = [token] + value
        return patient

    def save_vocab(self, dest: str):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

    def _patient_iterator(self, features: dict(str, list(list))):
        for i in range(len(features['concept'])):
            yield {key: values[i] for key, values in features.items()}

