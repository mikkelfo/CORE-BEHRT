from torch.utils.data import Dataset, IterableDataset
import torch
import pandas as pd
import numpy as np
from os.path import join

from typing import List, Tuple, Dict

class BaseDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.max_segments = self.get_max_segments()

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: torch.tensor(values[index]) for key, values in self.features.items()}

    def get_max_segments(self):
        if 'segment' not in self.features:
            return None
        return max([max(segment) for segment in self.features['segment']]) + 1


class MLMDataset(BaseDataset):
    def __init__(self, features: dict, vocabulary='vocabulary.pt', masked_ratio=0.3, ignore_special_tokens=True):
        super().__init__(features)
        if isinstance(vocabulary, str):
            self.vocabulary = self.load_vocabulary(vocabulary)
        elif isinstance(vocabulary, dict):
            self.vocabulary = vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')
        self.masked_ratio = masked_ratio
        if ignore_special_tokens:
            self.n_special_tokens = len([token for token in vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient['concept'] = masked_concepts
        patient['target'] = target
    
        return patient

    def _mask(self, patient: dict):
        concepts = patient['concept']

        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]      # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))                # Random number for each token
        masked = rng < self.masked_ratio                        # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]           # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)            # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8                                # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (adj_rng < 0.9)        # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary['[MASK]'], selected_concepts) # Replace with [MASK]
        selected_concepts = torch.where(rng_replace, torch.randint(self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:,0][masked]] = eligible_concepts[masked]    # Set "true" token
        masked_concepts[eligible_mask.nonzero()[:,0][masked]]= selected_concepts    # Sets new concepts

        return masked_concepts, target

    @staticmethod
    def load_vocabulary(vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')
    
class MLMLargeDataset(IterableDataset):
    def __init__(self, data_dir:str, mode:str,  **kwargs):
        """Initializes the dataset for masked language modeling
        mode is one of 'train', 'val' or 'test'"""
        self.kwargs = kwargs
        self.data_files = self.get_data_files(data_dir, mode)

        self.num_patients = len(torch.load(join(data_dir, f'{mode}_pids.pt')))
        self.vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))
        
        self.masked_ratio = self.kwargs.get('masked_ratio', 0.3)
        self.batch_size = kwargs.get('batch_size', 32)

        if kwargs.get('ignore_special_tokens', True):
            self.n_special_tokens = len([token for token in self.vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0

    def get_data_files(self, data_dir: str, mode: str):
        """Returns the data files for the given mode"""
        file_ids = torch.load(join(data_dir, f'{mode}_file_ids.pt'))
        return [join(data_dir, 'tokenized', f'tokenized_{mode}_{file_id}.pt') for file_id in file_ids]

    def __len__(self):
        """Number of patients in the dataset"""
        return self.num_patients

    def get_patient(self, file_name: str):
        """Loads a single patient from a file"""
        features = torch.load(file_name)
        num_patients = len(features['concept'])
        for patient_index in range(num_patients):
            patient = self.get_patient_dic(features, patient_index)
            masked_concepts, target = self._mask(patient)
            patient['concept'] = masked_concepts
            patient['target'] = target
            yield patient

    def get_patient_dic(self, features: Dict, patient_index: int):
        """Get a patient dictionary from a patient index"""
        return {key: torch.tensor(values[patient_index]) for key, values in features.items()}

    def __iter__(self):
        data_files = self.data_files.copy()  # Create a copy of data_files
        np.random.shuffle(data_files)  # Shuffle the copy
        for file_name in self.data_files:
            yield from self.get_patient(file_name)

    def _mask(self, patient: dict):
        concepts = patient['concept']
        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100
        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]      # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))                # Random number for each token
        masked = rng < self.masked_ratio                        # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]           # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)            # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8                                # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (adj_rng < 0.9)        # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)
        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary['[MASK]'], selected_concepts) # Replace with [MASK]
        selected_concepts = torch.where(rng_replace, torch.randint(self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant
        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:,0][masked]] = eligible_concepts[masked]    # Set "true" token
        masked_concepts[eligible_mask.nonzero()[:,0][masked]]= selected_concepts    # Sets new concepts

        return masked_concepts, target

    def save_vocabulary(self, file_name: str):
        torch.save(self.vocabulary, file_name)
    def save_pids(self, file_name: str):
        torch.save(self.pids, file_name)

class CensorDataset(BaseDataset):
    """
        n_hours can be both negative and positive (indicating before/after censor token)
        outcomes is a list of the outcome timestamps to predict
        censor_outcomes is a list of the censor timestamps to use
    """
    def __init__(self, features: dict, outcomes: list, censor_outcomes: list, n_hours: int):
        super().__init__(features)

        self.outcomes = outcomes
        self.censor_outcomes = censor_outcomes
        self.n_hours = n_hours

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        censor_timestamp = self.censor_outcomes[index]
        patient = self._censor(patient, censor_timestamp)
        patient['target'] = float(pd.notna(self.outcomes[index]))

        return patient

    def _censor(self, patient: dict, event_timestamp: float) -> dict:
        if pd.isna(event_timestamp):
            return patient
        else:
            # Only required when padding
            mask = patient['attention_mask']
            N_nomask = len(mask[mask==1])

            # Remove padding and replace background 0s with first non-background pos
            pos = patient['abspos'][:N_nomask]

            # censor the last n_hours
            dont_censor = (pos - event_timestamp - self.n_hours) <= 0

            # TODO: This removes padding as well - is this ok?
            for key, value in patient.items():
                patient[key] = value[dont_censor]

        return patient

