import pandas as pd
from datetime import datetime


class BaseCreator:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        return self.create(concepts, patients_info)


class AgeCreator(BaseCreator):
    feature = id = "age"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(
            patients_info["BIRTHDATE"].values, index=patients_info["PID"]
        ).to_dict()
        # Calculate approximate age
        ages = (
            ((concepts["TIMESTAMP"] - concepts["PID"].map(birthdates)).dt.days / 365.25)
            + 0.5
        ).round()

        concepts["AGE"] = ages
        return concepts


class AbsposCreator(BaseCreator):
    feature = id = "abspos"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        origin_point = datetime(**self.config.features.abspos)
        # Calculate hours since origin point
        abspos = (concepts["TIMESTAMP"] - origin_point).dt.total_seconds() / 60 / 60

        concepts["ABSPOS"] = abspos
        return concepts


class SegmentCreator(BaseCreator):
    feature = id = "segment"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        segments = concepts.groupby("PID")["ADMISSION_ID"].transform(
            lambda x: pd.factorize(x)[0] + 1
        )

        concepts["SEGMENT"] = segments
        return concepts


class BackgroundCreator(BaseCreator):
    id = "background"
    prepend_token = "BG_"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        info = patients_info.set_index("PID").to_dict("index")
        origin_point = datetime(**self.config.features.abspos)

        background = {"PID": [], "CONCEPT": []}
        background.update(
            {k.upper(): [] for k in self.config.features if k != "background"}
        )

        for pid in concepts.PID.unique():
            if pid in info:
                p_info = info[pid]
                for col in self.config.features.background:
                    if col in p_info and pd.notna(p_info[col]):
                        background["PID"].append(pid)
                        background["CONCEPT"].append(self.prepend_token + p_info[col])
                        background["ABSPOS"].append(
                            (p_info["BIRTHDATE"] - origin_point).total_seconds()
                            / 60
                            / 60
                        )

        if "segment" in self.config.features:
            background["SEGMENT"] = 0

        if "age" in self.config.features:
            background["AGE"] = -1

        # Prepend background to concepts
        background = pd.DataFrame(background)
        return pd.concat([background, concepts])


""" SIMPLE EXAMPLES """


class SimpleValueCreator(BaseCreator):
    id = "value"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        concepts["CONCEPT"] = concepts["CONCEPT"] + "_" + concepts["VALUE"].astype(str)

        return concepts


class QuartileValueCreator(BaseCreator):
    id = "quartile_value"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        quartiles = concepts.groupby("CONCEPT")["value"].transform(
            lambda x: pd.qcut(x, 4, labels=False)
        )
        concepts["CONCEPT"] = concepts["CONCEPT"] + "_" + quartiles.astype(str)

        return concepts
