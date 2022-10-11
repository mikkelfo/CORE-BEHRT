from torch.utils.data import Dataset, DataLoader
import random


# TODO: Also keep masked positions?
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
        masked_seq = [0]*N

        rng = random.random()
        
        for i in range(N):
            if rng < 0.8:           # 80% - Mask token
                masked_seq[i] = 0
            elif 0.8 < rng < 0.9:   # 10% - replace with random word
                masked_seq[i] = random.randint(1, max(self.vocab.values()))
            else:                   # 10% - Do nothing        
                masked_seq[i] = seq[i]   
        

        return self.codes[index], self.segments[index], masked_seq
