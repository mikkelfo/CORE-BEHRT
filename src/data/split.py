import os
import torch
import pandas as pd
from datetime import datetime
import src.common.utils as utils
import src.common.setup as setup


class Splitter:
    def __init__(self, config, dir=None, split_name="splits.pt") -> None:
        self.config = config
        self.dir = config.paths.extra_dir if dir is None else dir
        self.split_name = split_name

    def __call__(
        self,
        data: dict,
        mode: str,
        ratios: dict = {"train": 0.8, "val": 0.2},
        file=None,
    ):
        self.validate_ratios(ratios)  # Check ratios sum to 1
        if mode == "random":
            return self.random_split(data, ratios)
        elif mode == "covid":
            return self.covid_split(data, ratios)
        elif mode == "load":
            return self.load_splits(data, file=file)
        else:
            raise ValueError("Invalid mode")

    def load_splits(self, data, file):
        splits = torch.load(os.path.join(self.dir, file))

        return self.return_splits(data, splits)

    def random_split(self, data: dict, ratios: dict = {"train": 0.8, "val": 0.2}):
        N = self.get_number_of_patients(data)

        splits = self._split_indices(N, ratios)  # Get indices for each split

        return self.return_splits(data, splits)

    def covid_split(self, data: dict, ratios: dict = {"train": 0.8, "val": 0.2}):
        N = self.get_number_of_patients(data)

        # Covid split is all patients where their entire abspos is < the covid_split abspos
        covid_split = self.get_covid_split()
        normal_split = [i for i in range(N) if i not in covid_split]

        # Split the normal split (i.e. non-covid patients)
        splits = self._split_indices(len(normal_split), ratios)
        splits = {
            key: [normal_split[i] for i in split] for key, split in splits.items()
        }  # Adjust range idx to patient idx

        splits["test"] = covid_split

        return self.return_splits(data, splits)

    def _split_indices(self, N: int, ratios: dict = {"train": 0.8, "val": 0.2}):
        torch.manual_seed(0)

        indices = torch.randperm(N).tolist()
        splits = {}
        for name, ratio in ratios.items():
            N_split = round(N * ratio)
            splits[name] = indices[:N_split]
            indices = indices[N_split:]

        # Add remaining indices to last split - incase of rounding error
        if len(indices) > 0:
            splits[name].extend(indices)

        return splits

    def get_covid_split(self, outcomes=None, covid_date=datetime(2020, 6, 17)):
        if outcomes is None:
            outcomes = torch.load(
                os.path.join(self.config.paths.data_dir, "outcomes.pt")
            )
        origin_point = datetime(**self.config.features.abspos)
        covid_abspos = utils.calc_abspos(covid_date, origin_point)
        covid_split = [
            i for i, val in enumerate(outcomes["COVID"]) if val < covid_abspos
        ]
        torch.save(
            covid_split, os.path.join(self.config.paths.extra_dir, "covid_split.pt")
        )
        return covid_split

    def return_splits(self, data: dict, splits: dict) -> dict:
        self.save_splits(splits)
        return {
            set: {key: [values[s] for s in split] for key, values in data.items()}
            for set, split in splits.items()
        }

    def save_splits(self, splits):
        torch.save(splits, os.path.join(self.dir, self.split_name))
        N = sum([len(s) for s in splits.values()])
        print(
            f"Resulting split ratios: {[round(len(s) / N, 2) for s in splits.values()]}"
        )

    @staticmethod
    def validate_ratios(ratios: dict):
        ratio_sum = round(sum(ratios.values()), 5)
        if ratio_sum != 1:
            raise ValueError(f"Sum of ratios ({ratios}) != 1 ({ratio_sum})")

    @staticmethod
    def get_number_of_patients(data: dict):
        return len(next(iter(data.values())))

    def mimic_split(self):
        pids = setup.get_pids_with_exclusion(self.config)

        mimic_split = {key: [] for key in ["train", "eval", "test"]}
        for split_name in mimic_split:
            split_ids = pd.read_csv(f"data/mimic3/{split_name}-id.txt", header=None)[
                0
            ].tolist()
            split = [i for i, pid in enumerate(pids) if pid in set(split_ids)]
            mimic_split[split_name] = split

        torch.save(mimic_split, os.path.join(self.dir, self.split_name))
        return mimic_split

    @staticmethod
    def isolate_holdout(features: dict, test_split):
        if isinstance(test_split, str):
            test_split = torch.load(test_split)
        else:
            assert isinstance(test_split, list)

        test_split = set(test_split)
        normal_split = [
            i for i in range(len(features["concept"])) if i not in test_split
        ]

        non_holdout = {
            key: [values[i] for i in normal_split] for key, values in features.items()
        }
        holdout = {
            key: [values[i] for i in test_split] for key, values in features.items()
        }

        return non_holdout, holdout
