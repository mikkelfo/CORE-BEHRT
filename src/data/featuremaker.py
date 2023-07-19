import os
from datetime import datetime
import pandas as pd
import torch
from src.data.creators import BaseCreator


class FeatureMaker:
    def __init__(self, config):
        self.config = config

        self.features = {
            "concept": [],
        }

        self.order = {"concept": 0, "background": -1}
        self.creators = {
            creator.id: creator for creator in BaseCreator.__subclasses__()
        }
        self.pipeline = self.create_pipeline()

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        for creator in self.pipeline:
            concepts = creator(concepts, patients_info)

        features = self.create_features(concepts)

        return features

    def create_pipeline(self):
        # Pipeline creation
        pipeline = []
        for id in self.config.features:
            creator = self.creators[id](self.config)
            pipeline.append(creator)
            if getattr(creator, "feature", None) is not None:
                self.features[creator.feature] = []

        # Reordering
        pipeline_creators = [
            creator.feature for creator in pipeline if hasattr(creator, "feature")
        ]
        for feature, pos in self.order.items():
            if feature in pipeline_creators:
                creator = pipeline.pop(pipeline_creators.index(feature))
                pipeline.insert(pos, creator)

        return pipeline

    def create_features(self, concepts: pd.DataFrame) -> dict:
        PIDs = []
        # Add standard info
        for pid, patient in concepts.groupby("PID", sort=False):
            PIDs.append(pid)

            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())

        torch.save(
            PIDs, os.path.join(self.config.paths.extra_dir, "PIDs.pt")
        )  # Save PIDs for identification
        return self.features
