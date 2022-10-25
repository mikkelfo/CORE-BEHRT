from torch.utils.data import Dataset
import random


class MLMDataset(Dataset):
    def __init__(self, codes, segments, vocab):
        self.codes = codes
        self.segments = segments
        self.vocab = vocab

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):

        seq = self.codes[index]
        N = len(seq)
        masked_seq = [self.vocab['[CLS]']] + [0]*(N-1)
        masked_pos = [-1] * N           # -1 is auto-ignored in loss function
        
        for i in range(1, N):
            rng = random.random()
            if rng < 0.15:              # Select 15% of the tokens
                rng /= 0.15             # Fix ratio to 0-100 interval
                if rng < 0.8:           # 80% - Mask token
                    masked_seq[i] = self.vocab['[MASK]']
                elif 0.8 <= rng < 0.9:  # 10% - replace with random word
                    masked_seq[i] = random.randint(1, max(self.vocab.values()))
                else:                   # 10% - Do nothing        
                    masked_seq[i] = seq[i]
                masked_pos[i] = 0       # Unignore this token in loss function
            else:
                masked_seq[i] = seq[i]   
        

        return self.codes[index], self.segments[index], masked_seq, masked_pos
