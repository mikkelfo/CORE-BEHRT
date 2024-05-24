import torch
from transformers import BatchEncoding

from ehr2vec.data_fixes.handle import Handler
from ehr2vec.common.utils import iter_patients

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
        self.truncation = config.get('truncation', None)
        self.cutoffs = config.get('cutoffs', None)
        
    def __call__(self, features: dict, truncation=None)->BatchEncoding:
        truncation = self.truncation if truncation is None else truncation
        return self.batch_encode(features, truncation)

    def batch_encode(self, features: dict, truncation=512)->BatchEncoding:
        data = {key: [] for key in features}
        data['attention_mask'] = []

        for patient in iter_patients(features):
            patient = self.insert_special_tokens(patient)                   # Insert SEP and CLS tokens

            if truncation and (len(patient['concept']) > truncation):
                patient = self.truncate(patient, max_len=truncation)        # Truncate patient to max_len
            
            # Created after truncation for efficiency
            patient['attention_mask'] = [1] * len(patient['concept'])       # Initialize attention mask

            patient['concept'] = self.encode(patient['concept'])            # Encode concepts

            for key, value in patient.items():
                data[key].append(value)

        return BatchEncoding(data)

    def encode(self, concepts: list)->list:
        """Encode concepts to vocabulary ids"""
        if self.cutoffs:
            concepts = self.limit_concepts_length(concepts) # Truncate concepts to max_concept_length
        if self.new_vocab:
            for concept in concepts:
                if concept not in self.vocabulary:
                    self.vocabulary[concept] = len(self.vocabulary)

            encoded_sequence = [self.vocabulary.get(concept, self.vocabulary['[UNK]']) for concept in concepts]
        else:
            encoded_sequence = [self.vocabulary.get(concept, self.find_closest_ancestor(concept)) for concept in concepts]
        return encoded_sequence
    
    def find_closest_ancestor(self, concept)->int:
        """Find closest ancestor of concept in vocabulary"""
        while concept not in self.vocabulary and len(concept)>0:
            concept = concept[:-1]
        return self.vocabulary.get(concept, self.vocabulary['[UNK]'])
    # TODO: thinks what happens to short tokens that don't occur, should we instead look at closest node in the tree including siblings?

    @staticmethod
    def truncate(patient: dict, max_len: int)->dict:
        """Truncate patient to max_len"""
        # Find length of background sentence (+2 to include CLS token and SEP token)
        background_length = len([x for x in patient.get('concept', []) if x.startswith('BG_')]) + 2
        truncation_length = max_len - background_length
        
        # Do not start seq with SEP token (SEP token is included in background sentence)
        if patient['concept'][-truncation_length] == '[SEP]':
            truncation_length -= 1

        for key, value in patient.items():
            patient[key] = value[:background_length] + value[-truncation_length:]    # Keep background sentence + newest information

        if "segment" in patient:  # Re-normalize segments after truncation
            patient["segment"] = Handler.normalize_segments(patient["segment"])
        return patient

    def pad(self, features: dict,  max_len: int)->dict:
        """Pad sequences to max_len"""
        padded_data = {key: [] for key in features}
        for patient in iter_patients(features):
            difference = max_len - len(patient['concept'])

            for key, values in patient.items():
                token = self.vocabulary['[PAD]'] if key == 'concept' else 0
                padded_data[key].append(values + [token] * difference)

        return padded_data

    def insert_special_tokens(self, patient: dict)->dict:
        """Insert SEP and CLS tokens into patient"""
        if self.config.sep_tokens:
            if 'segment' not in patient:
                raise Exception('Cannot insert [SEP] tokens without segment information')
            patient = self.insert_sep_tokens(patient)

        if self.config.cls_token:
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
            token = '[CLS]' if key == 'concept' else values[0]          # Determine token value (CLS for concepts, 0 for rest)
            patient[key] = [token] + values
        return patient

    def limit_concepts_length(self, concepts: list)->list:
        """Truncate concepts to max_concept_length"""
        return [concept[:self.cutoffs.get(concept[0], None)] for concept in concepts]

    def save_vocab(self, dest: str)->None:
        torch.save(self.vocabulary, dest)

    def freeze_vocabulary(self)->None:
        self.new_vocab = False

