import os
import torch
from datetime import datetime


def get_covid_split(config, outcomes, covid_date=datetime(2020, 6, 17)):
    origin_point = datetime(**config.features.abspos)
    covid_abspos = (covid_date - origin_point).total_seconds() / 60 / 60
    covid_split = [i for i, val in enumerate(outcomes["COVID"]) if val < covid_abspos]
    torch.save(covid_split, os.path.join(config.paths.extra_dir, "covid_split.pt"))
    return covid_split


class Splitter:
    def __call__(
        self, data: dict, ratios: list = [0.8, 0.2], dir: str = "", load_splits=None
    ):
        if load_splits is None:
            return self.covid_split(data, ratios, dir=dir)
        else:
            return self.load_splits(data, dir=dir, file=load_splits)

    @staticmethod
    def load_splits(data, dir, file):
        splits = torch.load(os.path.join(dir, file))
        # Train, val, test is last
        for split in splits:
            yield {key: [values[s] for s in split] for key, values in data.items()}

    # Notice ratios is the train/val split as test split is fixed
    def covid_split(self, data: dict, ratios: list = [0.8, 0.2], dir: str = ""):
        N = len(next(iter(data.values())))  # Get length of first value in dict

        # Covid split is all patients where their entire abspos is < the covid_split abspos
        covid_split = torch.load(os.path.join(dir, "covid_split.pt"))
        normal_split = [i for i in range(N) if i not in covid_split]

        splits = self._split_indices(len(normal_split), ratios)
        splits = [[normal_split[i] for i in split] for split in splits]

        # We put the covid_split as the last split
        splits.append(covid_split)
        torch.save(splits, os.path.join(dir, "splits.pt"))
        print(f"Resulting split ratios: {[round(len(s) / N, 2) for s in splits]}")
        # Train, val, test is last
        for split in splits:
            yield {key: [values[s] for s in split] for key, values in data.items()}

    def random_split(self, data: dict, ratios: list = [0.7, 0.2, 0.1], dir: str = ""):
        if round(sum(ratios), 5) != 1:
            raise ValueError(f"Sum of ratios ({ratios}) != 1 ({round(sum(ratios), 5)})")

        N = len(next(iter(data.values())))  # Get length of first value in dict

        splits = self._split_indices(N, ratios)
        torch.save(splits, os.path.join(dir, "splits.pt"))
        print(f"Resulting split ratios: {[round(len(s) / N, 2) for s in splits]}")
        # Train, val, test is last
        for split in splits:
            yield {key: [values[s] for s in split] for key, values in data.items()}

    def _split_indices(self, N: int, ratios: list = [0.8, 0.2]):
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

        return splits
