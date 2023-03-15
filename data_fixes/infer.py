import pandas as pd

class Inferrer():
    def __init__(self):
        self.functions = {
            'ADMISSION_ID': self.infer_admission_id,
            'TIMESTAMP': self.infer_timestamps_from_admission_id
        }

    def __call__(self, df: pd.DataFrame):
        return self.infer(df)

    def infer(self, df: pd.DataFrame):
        for col in self.functions.keys():
            if col in df.columns:
                df[col] = self.functions[col](df)
        return df


    # Infer admission IDs (NaNs between identical IDs are inferred)
    def infer_admission_id(self, df: pd.DataFrame):
        bf = df.sort_values('PID')
        mask = bf['ADMISSION_ID'].fillna(method='ffill') != bf['ADMISSION_ID'].fillna(method='bfill')   # Find NaNs between similar admission IDs
        bf.loc[mask, 'ADMISSION_ID'] = bf.loc[mask, 'ADMISSION_ID'].map(lambda x: 'unq_') + list(map(str, range(mask.sum())))   # Assign unique IDs to non-inferred NaNs

        return bf['ADMISSION_ID'].fillna(method='ffill')  # Assign neighbour IDs to inferred NaNs


    # Infer timestamps (NaNs within identical admission IDs a related timestamp)
    def infer_timestamps_from_admission_id(self, df: pd.DataFrame, strategy="last"):
        if strategy == "last":
            return df.groupby('ADMISSION_ID')['TIMESTAMP'].fillna(method='ffill')

        elif strategy == "first":
            return df.groupby('ADMISSION_ID')['TIMESTAMP'].transform(lambda x: x.fillna(x.dropna().iloc[0]))

        elif strategy == "mean":
            return df.groupby('ADMISSION_ID')['TIMESTAMP'].transform(lambda x: x.fillna(x.mean()))

