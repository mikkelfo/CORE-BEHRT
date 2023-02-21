import pandas as pd
from datetime import datetime
import glob
import itertools

class BaseCreator():
    def __init__(self, config):
        self.config: dict = config

    def __call__(self, concepts):
        return self.create(concepts)

    def create(self):
        raise NotImplementedError

    def read_file(self, cfg, file_path) -> pd.DataFrame:
        file_path = f'{cfg.data_dir}/{file_path}'

        file_type = file_path.split(".")[-1]
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'parquet':
            return pd.read_parquet(file_path)

class ConceptCreator(BaseCreator):
    feature = 'concept'
    def create(self, concepts):
        # Get all concept files
        path = glob.glob('concept.*', root_dir=self.config.data_dir)

        # Filter out concepts files
        if self.config.get('concepts') is not None:
            path = [p for p in path if p.split('.')[1] in self.config.concepts]
        
        # Load concepts
        concepts = pd.concat([self.read_file(self.config, p) for p in path]).reset_index(drop=True)
        
        concepts['TIMESTAMP'] = pd.to_datetime(concepts['TIMESTAMP'].str.slice(stop=10))
        concepts = concepts.sort_values('TIMESTAMP')

        return concepts

class AgeCreator(BaseCreator):
    feature = 'age'
    def create(self, concepts):
        patients_info = self.read_file(self.config, 'patients_info.csv')
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days // 365.25

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    feature = 'abspos'
    def create(self, concepts):
        abspos = self.config.abspos
        origin_point = datetime(abspos.year, abspos.month, abspos.day)
        # Calculate days since origin point
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.days

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = 'segment'
    def create(self, concepts):
        # Infer NaNs in ADMISSION_ID
        concepts['ADMISSION_ID'] = self._infer_admission_id(concepts)

        segments = concepts.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0]+1)

        concepts['SEGMENT'] = segments
        return concepts

    def _infer_admission_id(self, df):
        bf = df.sort_values('PID')
        mask = bf['ADMISSION_ID'].fillna(method='ffill') != bf['ADMISSION_ID'].fillna(method='bfill')   # Find NaNs between similar admission IDs
        bf.loc[mask, 'ADMISSION_ID'] = bf.loc[mask, 'ADMISSION_ID'].map(lambda x: 'unq_') + list(map(str, range(mask.sum())))   # Assign unique IDs to non-inferred NaNs
        bf['ADMISSION_ID'] = bf['ADMISSION_ID'].fillna(method='ffill')  # Assign neighbour IDs to inferred NaNs
        
        return bf['ADMISSION_ID']

class BackgroundCreator(BaseCreator):
    feature = 'background'
    def create(self, concepts):
        patients_info = self.read_file(self.config, 'patients_info.csv')

        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.background),
            'CONCEPT': itertools.chain.from_iterable([patients_info[col].tolist() for col in self.config.background])
        }

        for feature in self.config.features:
            if feature != self.feature:
                background[feature.upper()] = 0

        background = pd.DataFrame(background)

        return pd.concat([background, concepts])
        
