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

        features = self.create_features(concepts, patients_info)

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

    def create_features(
        self, concepts: pd.DataFrame, patients_info: pd.DataFrame
    ) -> tuple:
        # Add outcomes if in config
        outcomes = {outcome: [] for outcome in self.config.outcomes}
        info_dict = patients_info.set_index("PID").to_dict("index")
        origin_point = datetime(**self.config.features.abspos)

        PIDs = []
        # Add standard info and outcomes
        for pid, patient in concepts.groupby("PID"):
            PIDs.append(pid)
            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())
            for outcome in self.config.outcomes:
                patient_outcome = info_dict[pid][f"OUTCOME_{outcome}"]
                if pd.isna(patient_outcome):
                    outcomes[outcome].append(patient_outcome)
                else:
                    outcomes[outcome].append(
                        (patient_outcome - origin_point).total_seconds() / 60 / 60
                    )

        torch.save(PIDs, "PIDs.pt")  # Save PIDs for identification
        return self.features, outcomes
