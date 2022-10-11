from torch.utils.data import Dataset, DataLoader


class MLMDataset(Dataset):
    def __init__(self, codes, segments):
        self.codes = codes
        self.segments = segments

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.codes[index], self.segments[index]
