from creators import ConceptCreator, AgeCreator, AbsposCreator, SegmentCreator, DemographicsCreator


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
        pipeline = [ConceptCreator(self.config)]
        self.features['CONCEPT'] = []

        if 'ages' in self.config:
            pipeline.append(AgeCreator(self.config))
            self.features['AGE'] = []
        if 'abspos' in self.config:
            pipeline.append(AbsposCreator(self.config))
            self.features['ABSPOS'] = []
        if 'segments' in self.config:
            pipeline.append(SegmentCreator(self.config))
            self.features['SEGMENT'] = []
        if 'demographics' in self.config:
            pipeline.append(DemographicsCreator(self.config))


        return pipeline

    def create_features(self, concepts):
        def add_to_features(patient):
            for feature in self.features.keys():
                self.features[feature].append(patient[feature].tolist())
        concepts.groupby('PID').apply(add_to_features)

        return self.features

