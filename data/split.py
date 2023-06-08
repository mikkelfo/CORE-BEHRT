import torch


class Splitter():
    def __call__(self, data: dict, ratios: list = [0.7, 0.2, 0.1]):
        return self.split(data, ratios)

    def split(self, data: dict, ratios: list = [0.7, 0.2, 0.1]):
        if round(sum(ratios), 5) != 1:
            raise ValueError(f'Sum of ratios ({ratios}) != 1 ({round(sum(ratios), 5)})')

        N = len(next(iter(data.values())))  # Get length of first value in dict

        splits = self._split_indices(N, ratios)

        for split in splits:
            yield {key: [values[s] for s in split] for key, values in data.items()}

    @staticmethod
    def _split_indices(N: int, ratios: list = [0.7, 0.2, 0.1]):
        torch.manual_seed(0)

        indices = torch.randperm(N)
        splits = []
        for ratio in ratios:
            N_split = round(N * ratio)
            splits.append(indices[:N_split])
            indices = indices[N_split:]

        # Add remaining indices to last split - incase of rounding error
        if len(indices) > 0:
            splits[-1] = torch.cat((splits[-1], indices))

        torch.save(splits, 'splits.pt')
        print(f'Resulting split ratios: {[round(len(s) / N, 2) for s in splits]}')
        return splits

