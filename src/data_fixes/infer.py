import pandas as pd


class Inferrer:
    def __init__(self, functions: list = ["ADMISSION_ID", "TIMESTAMP"]):
        function_dict = {
            "ADMISSION_ID": self.infer_admission_id,
            "TIMESTAMP": self.infer_timestamps_from_admission_id,
        }
        self.functions = {col: function_dict[col] for col in functions}

    def __call__(self, df: pd.DataFrame):
        return self.infer(df)

    def infer(self, df: pd.DataFrame):
        for col in self.functions:
            if col in df.columns:
                df[col] = self.functions[col](df)
        return df

    # Infer admission IDs (NaNs between identical IDs are inferred)
    @staticmethod
    def infer_admission_id(df: pd.DataFrame):
        bf = df.sort_values("PID")
        mask = bf["ADMISSION_ID"].fillna(method="ffill") != bf["ADMISSION_ID"].fillna(
            method="bfill"
        )  # Find NaNs between similar admission IDs
        bf.loc[mask, "ADMISSION_ID"] = bf.loc[mask, "ADMISSION_ID"].map(
            lambda _: "unq_"
        ) + list(
            map(str, range(mask.sum()))
        )  # Assign unique IDs to non-inferred NaNs

        return bf["ADMISSION_ID"].fillna(
            method="ffill"
        )  # Assign neighbour IDs to inferred NaNs

    # Infer timestamps (NaNs within identical admission IDs a related timestamp)
    @staticmethod
    def infer_timestamps_from_admission_id(df: pd.DataFrame, strategy="last"):
        if strategy == "last":
            return df.groupby("ADMISSION_ID", sort=False)["TIMESTAMP"].fillna(
                method="ffill"
            )

        elif strategy == "first":
            return df.groupby("ADMISSION_ID", sort=False)["TIMESTAMP"].transform(
                lambda x: x.fillna(x.dropna().iloc[0])
            )

        elif strategy == "mean":
            return df.groupby("ADMISSION_ID", sort=False)["TIMESTAMP"].transform(
                lambda x: x.fillna(x.mean())
            )
