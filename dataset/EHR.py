from torch.utils.data import Dataset
import random
import torch


class EHRDataset(Dataset):
    def __init__(self, data, vocabulary=None, masked=False, masked_ratio=0.3):
        self.data = data

        self.vocabulary = vocabulary
        self.masked = masked
        self.masked_ratio = masked_ratio

    def __len__(self):
        return len(self.data['events'])

    def __getitem__(self, index):
        '''
            Returns tuple of
                events, attention mask, None, None              if masked=False
                events, attention mask, masked events, target   if masked=True
        '''
        events, mask = self.data['events'][index], self.data['attention_mask'][index]
        if not self.masked:
            masked_events, target = torch.empty(0), torch.empty(0)
        else:
            N = len(events)
            N_nomask = len(mask[mask==1])
            
            masked_events = torch.clone(events)
            target = torch.ones(N) * -100
            target = torch.tensor([-100] * N)           # -100 is auto-ignored in loss function
            
            for i in range(1, N_nomask-1):              # Only mask non-PAD tokens
                rng = random.random()
                if rng < self.masked_ratio:             # Select set % of the tokens
                    rng /= self.masked_ratio            # Fix ratio to 0-100 interval
                    if rng < 0.8:                       # 80% - Mask token
                        masked_events[i][0] = self.vocabulary['[MASK]']
                    elif 0.8 <= rng < 0.9:              # 10% - replace with random word
                        masked_events[i][0] = random.randint(5, max(self.vocabulary.values()))
                    else:                               # 10% - Do nothing        
                        masked_events[i][0] = events[i][0]

                    target[i] = events[i][0]            # Set "true" token
            

        return events, mask, masked_events, target

    def split(self, test_ratio):
        N = len(self.data['events'])
        torch.manual_seed(0)
        indices = torch.randperm(N)
        N_test = int(N*test_ratio)

        train_indicies = indices[N_test:]
        test_indices = indices[:N_test]

        train_data = {}
        test_data = {}
        
        for key, value in self.data.items():
            train_data[key] = [value[i] for i in train_indicies]
            test_data[key] = [value[i] for i in test_indices]
        
        train_set = EHRDataset(train_data)
        test_set = EHRDataset(test_data)

        return train_set, test_set

    def setup_mlm(self, args):
        self.set_masked(True)
        self.set_vocabulary(args.vocabulary)
        self.set_masked_ratio(args.masked_ratio)

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
        return max([event[3] for patient in self.data['events'] for event in patient])
