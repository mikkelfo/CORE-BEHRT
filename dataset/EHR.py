from torch.utils.data import Dataset
import torch


class EHRDataset(Dataset):
    def __init__(self, features: dict[str, torch.LongTensor], vocabulary=None, masked=False, masked_ratio=0.3):
        self.features = features

        self.vocabulary = vocabulary
        self.masked = masked
        self.masked_ratio = masked_ratio

    def __len__(self):
        return self.data['concept']

    def __getitem__(self, index):
        patient = {key: values[index] for key, values in self.features.items()}
        if self.masked:
            masked_concepts, target = self._mask(patient)
            patient['masked_concepts'] = masked_concepts
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

    def split(self, ratios: list = [0.7, 0.1, 0.2]):
        if round(sum(ratios), 5) != 1:
            raise ValueError(f'Sum of ratios ({ratios}) != 1 ({round(sum(ratios), 5)})')
        torch.manual_seed(0)

        N = len(self.features['concept'])

        splits = self._split_indices(N, ratios)

        for split in splits:
            yield EHRDataset({key: values[split] for key, values in self.features.items()})

    def _split_indices(self, N: int, ratios: list):
        indices = torch.randperm(N)
        splits = []
        for ratio in ratios:
            N_split = round(N * ratio)
            splits.append(indices[:N_split])
            indices = indices[N_split:]

        # Add remaining indices to last split - incase of rounding error
        if len(indices) > 0:
            splits[-1] = torch.cat((splits[-1], indices))

        print(f'Resulting split ratios: {[round(len(s) / N, 2) for s in splits]}')
        return splits

    def setup_mlm(self, vocabulary, masked_ratio):
        self.set_masked(True)
        self.set_vocabulary(vocabulary)
        self.set_masked_ratio(masked_ratio)

    def set_masked(self, boolean: bool):
        self.masked = boolean

    def set_vocabulary(self, vocabulary):
        if isinstance(vocabulary, str):
            with open(vocabulary, 'rb') as f:
                self.vocabulary = torch.load(f)
        elif isinstance(vocabulary, dict):
            self.vocabulary = vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')

    def set_masked_ratio(self, ratio: float):
        self.masked_ratio = ratio

    def get_max_segments(self):
        if 'segment' not in self.data:
            raise ValueError('No segment data found. Please add segment data to dataset')
        return max([max(segment) for segment in self.data['segment']])

