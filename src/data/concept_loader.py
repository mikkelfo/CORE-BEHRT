import pandas as pd
from datetime import datetime
import glob
import dateutil
import os


class ConceptLoader:
    def __call__(
        self,
        concepts=["diagnose", "medication"],
        patients_info: str = "patients_info.csv",
        data_dir: str = "data/formatted",
    ):
        return self.load(
            concepts=concepts, data_dir=data_dir, patients_info=patients_info
        )

    def load(
        self,
        concepts=["diagnose", "medication"],
        patients_info: str = "patients_info.csv",
        data_dir: str = "data/formatted",
    ):
        # Get all concept files
        concept_paths = os.path.join(data_dir, "concept.*")
        path = glob.glob(concept_paths)

        # Filter out concepts files
        path = [p for p in path if p.split(".")[1] in concepts]

        # Load concepts
        concepts = pd.concat(
            [self._read_file(p) for p in path], ignore_index=True
        ).drop_duplicates()

        concepts = concepts.sort_values("TIMESTAMP")

        # Load patient data
        patient_path = os.path.join(data_dir, patients_info)
        patients_info = self._read_file(patient_path)

        return concepts, patients_info

    def _read_file(self, file_path: str) -> pd.DataFrame:
        file_type = file_path.split(".")[-1]
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "parquet":
            df = pd.read_parquet(file_path)

        for col in self.detect_date_columns(df):
            df[col] = self.fix_formatting_and_convert_to_dt(df, col)
            df[col] = df[col].dt.tz_localize(None)

        return df

    @staticmethod
    def detect_date_columns(df: pd.DataFrame):
        date_columns = []
        for col in df.columns:
            if isinstance(df[col], datetime):
                continue
            if "TIME" in col.upper() or "DATE" in col.upper():
                try:
                    first_non_na = df.loc[df[col].notna(), col].iloc[0]
                    dateutil.parser.parse(first_non_na)
                    date_columns.append(col)
                except:
                    continue
        return date_columns

    # Can add more fixes depending on datasets
    def fix_formatting_and_convert_to_dt(self, df, col):
        try:  # Check if formatting is needed
            return pd.to_datetime(df[col], errors="raise")
        except:
            for func in [
                self.double_punctuation,
                self.cutoff,
            ]:  # Do multiple attempts to fix formatting
                try:
                    df[col] = func(df[col])  # Accumulate fixes - TODO: Do we need this?
                    return pd.to_datetime(df[col], errors="raise")
                except:
                    continue

        return pd.to_datetime(
            df[col], errors="coerce"
        )  # Else: Invalid parsing will be set to NaT

    @staticmethod
    def double_punctuation(col):
        return col.str.replace(".:", ".", regex=False)

    @staticmethod
    def cutoff(col):
        return col.map(lambda x: x[:19] if isinstance(x, str) else x)
