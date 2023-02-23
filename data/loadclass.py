from creators import BaseCreator


class FeatureMaker():
    def __init__(self, config):
        self.config = config

        self.features = {}

        self.pipeline = self.create_pipeline()

        self.order = {
            'concept': 0,
            'background': -1
        }

    def __call__(self):
        concepts = None
        for creator in self.pipeline:
            concepts = creator(concepts)

        features = self.create_features(concepts)

        return features
    
    def create_pipeline(self):
        creators = {creator.feature: creator for creator in BaseCreator.__subclasses__()}

        # Pipeline creation
        pipeline = []
        for feature in self.config.features:
            pipeline.append(creators[feature](self.config))
            if feature != 'background':
                self.features.setdefault(feature, [])

        # Reordering
        pipeline_creators = [creator.feature for creator in pipeline]
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

