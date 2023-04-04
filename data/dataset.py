from torch.utils.data import Dataset
import torch
import pandas as pd


class BaseDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.max_segments = self.get_max_segments()

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.features.items()}

    def get_max_segments(self):
        if 'segment' not in self.features:
            raise ValueError('No segment data found. Please add segment data to dataset')
        return max([max(segment) for segment in self.features['segment']]) + 1


class MLMDataset(BaseDataset):
    def __init__(self, features: dict, vocabulary: dict, masked_ratio=0.3, ignore_special_tokens=True):
        super().__init__(features)

        self.vocabulary = vocabulary
        self.masked_ratio = masked_ratio
        if ignore_special_tokens:
            self.n_special_tokens = 5   # TODO: Find a dynamic way to get this

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient['concept'] = masked_concepts
        patient['target'] = target
    
        return patient

    def _mask(self, patient: dict):
        concepts = torch.tensor(patient['concept'])

        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligble_mask = masked_concepts >= self.n_special_tokens
        eligble_concepts = masked_concepts[eligble_mask]        # Ignore special tokens
        rng = torch.rand(len(eligble_concepts))                 # Random number for each token
        masked = rng < self.masked_ratio                        # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligble_concepts[masked]            # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)            # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8                                # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (adj_rng < 0.9)        # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(rng_mask, self.vocabulary['[MASK]'], selected_concepts) # Replace with [MASK]
        selected_concepts = torch.where(rng_replace, torch.randint(self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs
        target[eligble_mask][masked] = eligble_concepts[masked] # Set "true" token
        eligble_concepts[masked] = selected_concepts            # Set masked token in concepts (masked_concepts is updated by eligble_concepts)

        return masked_concepts, target

    def load_vocabulary(self, vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')


class CensorDataset(BaseDataset):
    """
        Uses .index, so it finds first occurence of censor token
        Takes censor_token as int (post-tokenizer)
        n_hours can be both negative and positive (indicating before/after censor token)
    """
    def __init__(self, features: dict, censor_token: int, n_hours: int, outcomes: dict):
        super().__init__(features)

        self.censor_token = censor_token
        self.n_hours = n_hours
        self.outcomes = outcomes

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        patient['target'] = pd.isna(self.outcomes[index])   # TODO: Boolean target or int?

        return self._censor(patient)

    def _censor(self, patient: dict) -> dict:
        # Find first occurence of censor token
        token_idx = patient['concept'].index(self.censor_token)

        # Only required when padding
        mask = torch.tensor(patient['attention_mask'])
        N_nomask = len(mask[mask==1])

        # Remove padding 
        pos = patient['abspos'][:N_nomask]

        # censor the last n_hours
        dont_censor = (pos - pos[token_idx] - self.n_hours) <= 0    # Include n_hours or not? (<= or <)

        # TODO: This removes padding as well - is this ok?
        for key, value in patient.items():
            patient[key] = value[dont_censor]

        return patient


