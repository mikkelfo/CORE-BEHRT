from torch.utils.data import Dataset
import torch
from typing import Dict, List
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

    

class H_MLM_Dataset(BaseDataset):
    def __init__(self, features, **kwargs):
        super().__init__(features, **kwargs)
        self.h_vocabulary = self.load_vocabulary(self.kwargs.get('h_vocabulary', 'h_vocabulary.pt'))
        
    def __getitem__(self, index):
        """
        return: dictionary with codes for patient with index 
        """ 
        patient = super().__getitem__(index)
        mask = self.get_mask(patient)
        patient['attention_mask'] = mask
        ids, labels = self.random_mask_ids(patient['idx']) 
        patient['values'] = self.mask_values(ids, patient['values']) # otherwise model could infer e.g. lab test
        # pad code sequence, segments and label
        #patient['codes'] = codes
        patient['idx'] = ids
        patient['labels'] = labels
        for channel in self.channels+['labels', 'idx']:
            out_dic[channel] = self.seq_padding(patient[channel], 
                self.pad_tokens[channel])    
            #print(channel, out_dic[channel])    
            out_dic[channel] = torch.LongTensor(out_dic[channel])    
            
        return out_dic

    def __len__(self):
        return len(self.data)
    
    def mask_values(self, ids, values):
        """Mask values the same way ids were masked"""
        mask_id = self.vocab['[MASK]']
        mask = np.array(ids)==mask_id
        values = np.array(values)
        values[mask] = 1
        return values.tolist()

    def get_mask(self, patient):
        mask = np.ones(self.pad_len)
        mask[len(patient['codes']):] = 0
        return mask

    def init_pad_len(self, data, pad_len):
        if isinstance(pad_len, type(None)):
            lens = np.array([len(d['codes']) for d in data])
            self.pad_len = int(np.max(lens)) 
        else:
            self.pad_len = pad_len

    def init_nonspecial_ids(self):
        """We use by default < as special sign for special tokens"""
        self.nonspecial_ids = [v for k,v in self.vocab.items() if not k.startswith('[')]
        
    def seq_padding(self, seq, pad_token):
        """Pad a sequence to the given length."""
        return seq + (self.pad_len-len(seq)) * [pad_token]

    def random_mask_ids(self, ids, seed=0):
        """mask code with 15% probability, 80% of the time replace with [MASK], 
            10% of the time replace with random token, 10% of the time keep original"""
        rng = default_rng(seed)
        masked_ids = ids.copy()
        labels = len(ids) * [-100] 
        for i, id in enumerate(ids):
            if id not in self.nonspecial_ids:
                continue
            prob = rng.uniform()
            if prob<self.mask_prob:
                prob = rng.uniform()  
                # 80% of the time replace with [MASK] 
                if prob < 0.8:
                    masked_ids[i] = self.vocab['[MASK]']
                # 10% change token to random token
                elif prob < 0.9:      
                    masked_ids[i] = rng.choice(self.nonspecial_ids) 
                # 10% keep original
                labels[i] = id
        return masked_ids, labels

