import pandas as pd
from datetime import datetime
import glob
import dateutil
import os


class ConceptLoader():
    def __init__(self, concepts: list = None):
        self.concepts = ['diagnose', 'medication'] if concepts is None else concepts

    def __call__(self, data_dir: str = 'formatted_data'):
        return self.load(data_dir)
    
    def load(self, data_dir: str = 'formatted_data'):
        # Get all concept files
        concept_paths = os.path.join(data_dir, 'concept.*')
        path = glob.glob(concept_paths)

        # Filter out concepts files
        path = [p for p in path if p.split('.')[1] in self.concepts]
        
        # Load concepts
        concepts = pd.concat([self._read_file(p) for p in path], ignore_index=True).drop_duplicates()
        
        concepts = concepts.sort_values('TIMESTAMP')

        # Load patient data
        patient_path = os.path.join(data_dir, 'patients_info.csv')
        patients_info = self._read_file(patient_path)

        return concepts, patients_info

    def _read_file(self, file_path: str) -> pd.DataFrame:
        file_type = file_path.split(".")[-1]
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)

        for col in self._detect_date_columns(df):
            df[col] = df[col].apply(lambda x: x[:10] if isinstance(x, str) else x)
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.tz_localize(None)

        return df

    def _detect_date_columns(self, df: pd.DataFrame):
        date_columns = []
        for col in df.columns:
            if isinstance(df[col], datetime):
                continue
            if 'TIME' in col.upper() or 'DATE' in col.upper():
                try:
                    first_non_na = df.loc[df[col].notna(), col].iloc[0]
                    dateutil.parser.parse(first_non_na)
                    date_columns.append(col)
                except:
                    continue
        return date_columns

