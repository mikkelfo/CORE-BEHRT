from creators import BaseCreator, ConceptCreator, BackgroundCreator


class FeatureMaker():
    def __init__(self, config):
        self.config = config

        self.features = {}

        self.pipeline = self.create_pipeline()

    def __call__(self):
        concepts = None
        for creator in self.pipeline:
            concepts = creator(concepts)

        features = self.create_features(concepts)

        return features
    
    def create_pipeline(self):
        # Start pipeline with Concepts (always)
        pipeline = [ConceptCreator(self.config)]
        self.features['CONCEPT'] = []

        creators = BaseCreator.__subclasses__()

        # Add all other features (age, abspos, segment, etc.)
        for creator in creators:
            if creator.feature.lower() in self.config.features:
                pipeline.append(creator(self.config))
                self.features[creator.feature] = []

        # Last (optional) step is to create background - not its own feature
        if 'background' in self.config.features:
            pipeline.append(BackgroundCreator(self.config))

        return pipeline

    def create_features(self, concepts):
        def add_to_features(patient):
            for feature, value in self.features.items():
                value.append(patient[feature].tolist())
        concepts.groupby('PID').apply(add_to_features)

        return self.features

