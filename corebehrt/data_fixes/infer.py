import pandas as pd

""" Inferring is doing during ehr_preprocess """
class Inferrer:
    def __init__(self, functions: list = ['SEGMENT', 'TIMESTAMP']):
        function_dict = {
            'SEGMENT': self.infer_admission_id,
            'TIMESTAMP': self.infer_timestamps_from_admission_id
        }
        self.functions = {col: function_dict[col] for col in functions}

    def __call__(self, df: pd.DataFrame)->pd.DataFrame:
        return self.infer(df)

    def infer(self, df: pd.DataFrame)->pd.DataFrame:
        """Infer missing values in the dataframe."""
        for col in self.functions:
            if col in df.columns:
                df[col] = self.functions[col](df)
        return df

    # Infer admission IDs (NaNs between identical IDs are inferred)
    @staticmethod
    def infer_admission_id(df: pd.DataFrame)->pd.Series:
        """Infer admission IDs (NaNs between identical IDs are inferred)"""
        bf = df.sort_values('PID')
        mask = bf['SEGMENT'].fillna(method='ffill') != bf['SEGMENT'].fillna(method='bfill')   # Find NaNs between similar admission IDs
        bf.loc[mask, 'SEGMENT'] = bf.loc[mask, 'SEGMENT'].map(lambda _: 'unq_') + list(map(str, range(mask.sum())))   # Assign unique IDs to non-inferred NaNs

        return bf['SEGMENT'].fillna(method='ffill')  # Assign neighbour IDs to inferred NaNs

    # Infer timestamps (NaNs within identical admission IDs a related timestamp)
    @staticmethod
    def infer_timestamps_from_admission_id(df: pd.DataFrame, strategy="last")->pd.Series:
        if strategy == "last":
            return df.groupby('SEGMENT')['TIMESTAMP'].fillna(method='ffill')

        elif strategy == "first":
            return df.groupby('SEGMENT')['TIMESTAMP'].transform(lambda x: x.fillna(x.dropna().iloc[0]))

        elif strategy == "mean":
            return df.groupby('SEGMENT')['TIMESTAMP'].transform(lambda x: x.fillna(x.mean()))

