import pandas as pd
import torch
from data.creators import BaseCreator
from data.utils import Utilities


class FeatureMaker():
    def __init__(self, config):
        self.config = config

        self.features = {
            'concept': [],
        }

        self.order = {
            'concept': 0,
            'background': -1
        }
        self.creators = {creator.id: creator for creator in BaseCreator.__subclasses__() if creator.id in self.config.keys()}
        self.pipeline = self.create_pipeline()
        

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)->dict:
        for creator in self.pipeline:
            concepts = creator(concepts, patients_info)
            concepts['CONCEPT'] = concepts['CONCEPT'].astype(str)
        features = self.create_features(concepts, patients_info)

        return features
    
    def create_pipeline(self)->list:
        """Create the pipeline of feature creators."""
        # Pipeline creation
        pipeline = []
        for id in self.config:
            creator = self.creators[id](self.config)
            pipeline.append(creator)
            if getattr(creator, 'feature', None) is not None:
                self.features[creator.feature] = []

        # Reordering
        pipeline_creators = [creator.feature for creator in pipeline if hasattr(creator, 'feature')]
        for feature, pos in self.order.items():
            if feature in pipeline_creators:
                creator = pipeline.pop(pipeline_creators.index(feature))
                pipeline.insert(pos, creator)

        return pipeline

    def create_features(self, concepts: pd.DataFrame, patients_info: pd.DataFrame) -> tuple:
        # Add standard info
        pids = []
        for pid, patient in concepts.groupby('PID'):
            pids.append(pid)
            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())
        # Add outcomes if in config
        
        info_dict = patients_info.set_index('PID').to_dict('index')
        
        # Add outcomes
        if hasattr(self.config, 'outcomes'):
            outcomes = []
            for pid, patient in concepts.groupby('PID'):
                for outcome in self.config.outcomes:
                    patient_outcome = info_dict[pid][f'{outcome}']
                    if pd.isna(patient_outcome):
                        outcome_abspos = torch.inf
                    else:
                        outcome_abspos = Utilities.get_abspos_from_origin_point([patient_outcome], self.config.abspos)[0]
                    outcomes.append(outcome_abspos)

            return self.features, outcomes
        else:
            return self.features, pids

