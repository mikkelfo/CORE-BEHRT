import torch
from transformers import BatchEncoding

from corebehrt.common.utils import iter_patients


class EHRTokenizer:
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
        self.cutoffs = config.get('cutoffs')
        
    def __call__(self, features: dict)->BatchEncoding:
        return self.batch_encode(features)

    def batch_encode(self, features: dict)->BatchEncoding:
        data = {key: [] for key in features}
        data['attention_mask'] = []

        for patient in iter_patients(features):
            patient = self.insert_special_tokens(patient)                   # Insert SEP and CLS tokens
            
            # Created after truncation for efficiency
            patient['attention_mask'] = [1] * len(patient['concept'])       # Initialize attention mask

            patient['concept'] = self.encode(patient['concept'])            # Encode concepts

            for key, value in patient.items():
                data[key].append(value)

        return BatchEncoding(data)

    def encode(self, concepts: list)->list:
        """Encode concepts to vocabulary ids"""
        if self.cutoffs:
            concepts = self.limit_concepts_length(concepts) # Cutoff concepts to max_concept_length
        if self.new_vocab:
            for concept in concepts:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

        encoded_sequence = [self.vocabulary.get(concept, self.vocabulary['[UNK]']) for concept in concepts]
        return encoded_sequence

    def insert_special_tokens(self, patient: dict)->dict:
        """Insert SEP and CLS tokens into patient"""
        patient = self.insert_sep_tokens(patient)
        patient = self.insert_cls_token(patient)
        
        return patient

    @staticmethod
    def insert_sep_tokens(patient: dict)->dict:
        """Insert SEP tokens into patient"""
        padded_segment = patient['segment'] + [None]                # Add None to last entry to avoid index out of range

        for key, values in patient.items():
            new_seq = []
            for i, val in enumerate(values):
                new_seq.append(val)

                if padded_segment[i] != padded_segment[i+1]:
                    token = '[SEP]' if key == 'concept' else val
                    new_seq.append(token)

            patient[key] = new_seq

        return patient

    @staticmethod
    def insert_cls_token(patient: dict)->dict:
        """Insert CLS token into patient"""
        for key, values in patient.items():
            token = '[CLS]' if key == 'concept' else values[0]          # Determine token value (CLS for concepts, 1st value for rest)
            patient[key] = [token] + values
        return patient

    def limit_concepts_length(self, concepts: list)->list:
        """Cutoff concepts to max_concept_length"""
        return [concept[:self.cutoffs.get(concept[0])] for concept in concepts]

    def save_vocab(self, dest: str)->None:
        torch.save(self.vocabulary, dest)

    def freeze_vocabulary(self)->None:
        self.new_vocab = False

