import torch
from base import BaseDataset


class MLMDataset(BaseDataset):
    def __init__(self, features: dict[str, torch.LongTensor], **kwargs):
        super().__init__(features, **kwargs)

        self.vocabulary = self.load_vocabulary(self.kwargs['vocabulary'])
        self.masked_ratio = self.kwargs.get('masked_ratio', 0.3)

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        patient['concept'] = masked_concepts
        patient['target'] = target
    
        return patient

    def _mask(self, patient: dict[str, torch.LongTensor]):
        concepts = patient['concept']
        mask = patient['attention_mask']

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

    def load_vocabulary(self, vocabulary):
        if isinstance(vocabulary, str):
            with open(vocabulary, 'rb') as f:
                self.vocabulary = torch.load(f)
        elif isinstance(vocabulary, dict):
            self.vocabulary = vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')

