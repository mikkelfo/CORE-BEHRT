from torch.utils.data import Dataset
import torch
from typing import Dict, List, Tuple
import numpy as np



class BaseDataset(Dataset):
    def __init__(self, features: dict, **kwargs):
        self.features = features
        self.kwargs = kwargs
        self.max_segments = self.get_max_segments()

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.features.items()}

    def get_max_segments(self):
        if 'segment' not in self.features:
            raise ValueError('No segment data found. Please add segment data to dataset')
        return max([max(segment) for segment in self.features['segment']]) + 1

    def load_vocabulary(self, vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')

class MLMDataset(BaseDataset):
    def __init__(self, features: dict, **kwargs):
        super().__init__(features, **kwargs)

        self.vocabulary = self.load_vocabulary(self.kwargs.get('vocabulary', 'vocabulary.pt'))
        self.masked_ratio = self.kwargs.get('masked_ratio', 0.3)

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient['concept'] = masked_concepts
        patient['target'] = target
    
        return patient

    def _mask(self, patient: dict):
        concepts = torch.tensor(patient['concept'])
        mask = torch.tensor(patient['attention_mask'])

        N = len(concepts)
        N_nomask = len(mask[mask==1])

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligble_concepts = masked_concepts[1:N_nomask-1]        # We dont mask CLS and last SEP token
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
        selected_concepts = torch.where(rng_replace, torch.randint(5, len(self.vocabulary), (len(selected_concepts),)), selected_concepts) # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)   # Redundant

        # Update outputs
        target[1:N_nomask-1][masked] = eligble_concepts[masked] # Set "true" token
        eligble_concepts[masked] = selected_concepts            # Set masked token in concepts (masked_concepts is updated by eligble_concepts)

        return masked_concepts, target

class H_MLMDataset(MLMDataset):
    def __init__(self, features:Dict[str,List], seed:int=0, **kwargs):
        super().__init__(features, **kwargs)
        self.h_vocabulary = self.load_vocabulary(self.kwargs.get('h_vocabulary', 'h_vocabulary.pt'))
        self.leaf_nodes = self.load_leaf_nodes(self.kwargs.get('leaf_nodes', 'leaf_nodes.pt'))
        self.base_leaf_probs = self.load_leaf_probs(self.kwargs.get('base_leaf_probs', 'base_leaf_probs.pt'))
        self.default_rng = np.random.default_rng(seed)
        self.mask_sep = False
        if 'mask_sep' in kwargs:
            self.mask_sep = kwargs['mask_sep']
        self.special_ids = [v for k,v in self.vocabulary.items() if '[' in k]
        if self.mask_sep:
            self.special_ids.remove(self.vocabulary['[SEP]'])

    def __getitem__(self, index:int):
        """
        return: dictionary with codes for patient with index 
        """ 
        patient = {key: values[index] for key, values in self.features.items()}
        masked_concepts, targets = self.random_mask(patient) 

        patient['target'] = targets
        patient['concept'] = masked_concepts
        for k, v in patient.items():
            if k in ['age', 'abspos']:
                dtype = torch.float
            else:
                dtype = torch.long
            patient[k] = torch.tensor(v, dtype=dtype)
        return patient
    # TODO: make sure at least one concept is masked per sequence
    def random_mask(self, patient:Dict[str,List])->Tuple[List,List]:
        """mask code with certain probability, 80% of the time replace with [MASK], 
            10% of the time replace with random token, 10% of the time keep original"""
        
        concepts, targets = patient['concept'], patient['target']

        masked_concepts = concepts
        masked_targets = len(concepts) * [(-100,)*len(targets[0])] # -100 is ignored in loss function

        for i, concept, target in zip(range(len(concepts)), concepts, targets):
            if concept in self.special_ids: # dont mask special tokens, if SEP token should be masked, it's excluded from special_ids
                continue
            if i==len(concepts)-1 and concept==self.vocabulary['[SEP]']: # dont mask last sep token
                continue
            prob = self.default_rng.uniform()
            if prob<self.masked_ratio:
                prob /= self.masked_ratio
                # 80% of the time replace with [MASK] 
                if prob < 0.8:
                    masked_concepts[i] = self.vocabulary['[MASK]']
                # 10% change token to random token
                elif prob < 0.9:      
                    masked_concepts[i] = self.default_rng.choice(list(self.vocabulary.values()))
                # 10% keep original 
                masked_targets[i] = target
        
        return masked_concepts, masked_targets

    def load_leaf_nodes(self, leaf_nodes):
        if isinstance(leaf_nodes, str):
            return torch.load(leaf_nodes)
        elif isinstance(leaf_nodes,  np.ndarray):
            return torch.from_numpy(leaf_nodes)
        elif isinstance(leaf_nodes, torch.Tensor):
            return leaf_nodes
        elif isinstance(leaf_nodes, list):
            return torch.tensor(leaf_nodes, dtype=torch.long)
        else:
            raise TypeError(f'Unsupported leaf_nodes/leaf_probs input {type(leaf_nodes)}')
 
    def load_leaf_probs(self, leaf_probs):
        return self.load_leaf_nodes(leaf_probs)