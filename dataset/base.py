from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):
    def __init__(self, features: dict[str, torch.LongTensor], **kwargs):
        self.features = features
        self.kwargs = kwargs

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.features.items()}

    def split(self, ratios: list = [0.7, 0.1, 0.2]):
        if round(sum(ratios), 5) != 1:
            raise ValueError(f'Sum of ratios ({ratios}) != 1 ({round(sum(ratios), 5)})')
        torch.manual_seed(0)

        N = len(self.features['concept'])

        splits = self._split_indices(N, ratios)

        for split in splits:
            yield self.__class__({key: values[split] for key, values in self.features.items()}, **self.kwargs)

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

    def get_max_segments(self):
        if 'segment' not in self.data:
            raise ValueError('No segment data found. Please add segment data to dataset')
        return max([max(segment) for segment in self.data['segment']])

