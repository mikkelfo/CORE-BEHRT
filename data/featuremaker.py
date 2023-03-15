from data.creators import BaseCreator


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
        self.creators = {creator.id: creator for creator in BaseCreator.__subclasses__()}
        self.pipeline = self.create_pipeline()

    def __call__(self, concepts, patients_info):
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
            if getattr(creator, 'feature', None) is not None:
                self.features[creator.feature] = []

        # Reordering
        pipeline_creators = [creator.feature for creator in pipeline if hasattr(creator, 'feature')]
        for feature, pos in self.order.items():
            if feature in pipeline_creators:
                creator = pipeline.pop(pipeline_creators.index(feature))
                pipeline.insert(pos, creator)

        return pipeline

    def create_features(self, concepts):
        def add_to_features(patient):
            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())
        concepts.groupby('PID').apply(add_to_features)

        return self.features

