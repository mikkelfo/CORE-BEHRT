import pandas as pd
from datetime import datetime
import glob
import itertools

class BaseCreator():
    def __init__(self, config):
        self.config: dict = config

    def __call__(self, concepts=None):
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
    feature = 'CONCEPT'
    def create(self, concepts):
        # Get all concept files
        path = glob.glob('concept.*', root_dir=self.config.data_dir)

        # Filter out concepts files
        if self.config.get('concepts') is not None:
            path = [p for p in path if p.split('.')[1] in self.config.concepts]
        
        # Load concepts
        concepts = pd.concat([self.read_file(self.config, p) for p in path]).reset_index(drop=True)
        
        concepts = concepts.sort_values('TIMESTAMP')
        concepts['TIMESTAMP'] = pd.to_datetime(concepts['TIMESTAMP'].str.slice(stop=10))

        return concepts

class AgeCreator(BaseCreator):
    feature = 'AGE'
    def create(self, concepts):
        patients_info = self.read_file(self.config, 'patients_info.csv')
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (concepts['TIMESTAMP'] - concepts['Key.Patient'].map(birthdates)).dt.days // 365.25

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    feature = 'ABSPOS'
    def create(self, concepts):
        year, month, day = self.config.abspos['year'], self.config.abspos['month'], self.config.abspos['day']
        origin_point = datetime(year, month, day)
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.days

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = 'SEGMENT'
    def create(self, concepts):
        concepts['ADMISSION_ID'] = self._infer_admission_id(concepts)

        segments = concepts.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0]+1)

        concepts['SEGMENT'] = segments
        return concepts

    def _infer_admission_id(self, df):
        bf = df.sort_values('PID')
        mask = bf['ADMISSION_ID'].fillna(method='ffill') != bf['ADMISSION_ID'].fillna(method='bfill')
        bf.loc[mask, 'ADMISSION_ID'] = bf.loc[mask, 'ADMISSION_ID'].map(lambda x: 'unq_') + list(map(str, range(mask.sum())))
        bf['ADMISSION_ID'] = bf['ADMISSION_ID'].fillna(method='ffill')
        
        return bf['ADMISSION_ID']

class DemographicsCreator(BaseCreator):
    feature = 'BACKGROUND'
    def create(self, concepts):
        patients_info = self.read_file(self.config, 'patients_info.csv')
        columns = self.config.demographics

        background = {
            'PID': patients_info['PID'].tolist() * len(columns),
        }
        background['CONCEPT'] = itertools.chain.from_iterable([patients_info[col].tolist() for col in columns])
        if self.config.ages: background['AGE'] = 0
        if self.config.abspos: background['ABSPOS'] = 0
        if self.config.segments: background['SEGMENT'] = 0

        background = pd.DataFrame(background)

        return pd.concat([background, concepts])
        
