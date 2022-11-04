from torch.utils.data import Dataset
import random


class EHRDataset(Dataset):
    def __init__(self, codes, segments, attention_mask, vocabulary=None, masked=False):
        self.codes = codes
        self.segments = segments
        self.attention_mask = attention_mask

        self.vocabulary = vocabulary
        self.masked = masked

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        '''
            Returns tuple of
                (outputs, None)             if masked=False
                (outputs, masked_outputs)   if masked=True
        '''
        if not self.masked:
            return (self.codes[index], self.segments[index], self.attention_mask[index]), (None, None)

        else:
            seq = self.codes[index]
            N = len(seq)
            masked_seq = [self.vocabulary['[CLS]']] + [0]*(N-2) + [self.vocabulary['[SEP]']]
            target = [-100] * N             # -100 is auto-ignored in loss function
            
            for i in range(1, N-1):
                rng = random.random()
                if rng < 0.15:              # Select 15% of the tokens
                    rng /= 0.15             # Fix ratio to 0-100 interval
                    if rng < 0.8:           # 80% - Mask token
                        masked_seq[i] = self.vocabulary['[MASK]']
                    elif 0.8 <= rng < 0.9:  # 10% - replace with random word
                        masked_seq[i] = random.randint(5, max(self.vocabulary.values()))
                    else:                   # 10% - Do nothing        
                        masked_seq[i] = seq[i]

                    target[i] = seq[i]      # Set "true" token

                else:
                    masked_seq[i] = seq[i]
            

            return (self.codes[index], self.segments[index], self.attention_mask[index]), (masked_seq, target)
