"""Feature engineering and preprocessing for the Titanic dataset.

The TitanicPreprocessor encapsulates every transformation applied before
the data reaches the model, so the exact same logic (fitted on the training
split) can be reused at inference time in the Streamlit app.
"""

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch", "FarePerPerson", "TicketGroupSize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck", "SexPclass"]

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
    "Dona": "Rare",
}


def _extract_title(name: str) -> str:
    match = re.search(r",\s*([^.]+)\.", str(name))
    title = match.group(1).strip() if match else "Unknown"
    return TITLE_MAP.get(title, title)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the raw Titanic columns. Does not impute or scale."""
    df = df.copy()
    df["Title"] = df["Name"].apply(_extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].apply(lambda c: str(c)[0] if pd.notna(c) else "U")
    df["SexPclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
    return df


class TitanicPreprocessor:
    """Fit on the training split, then transform train/val/inference data identically."""

    def __init__(self):
        self.age_medians_ = None
        self.fare_median_ = None
        self.embarked_mode_ = None
        self.scaler_ = StandardScaler()
        self.feature_columns_ = None

    def fit(self, df: pd.DataFrame):
        df = engineer_features(df)
        self.age_medians_ = df.groupby(["Title", "Pclass"])["Age"].median()
        self.global_age_median_ = df["Age"].median()
        self.fare_median_ = df["Fare"].median()
        self.embarked_mode_ = df["Embarked"].mode()[0]
        self.ticket_counts_ = df["Ticket"].value_counts().to_dict()

        transformed = self._apply_imputation_and_dummies(df)
        self.feature_columns_ = transformed.columns.tolist()
        self.scaler_.fit(transformed[NUMERIC_FEATURES])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(df)
        transformed = self._apply_imputation_and_dummies(df)
        transformed = transformed.reindex(columns=self.feature_columns_, fill_value=0)
        transformed[NUMERIC_FEATURES] = self.scaler_.transform(
            transformed[NUMERIC_FEATURES]
        )
        return transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def _apply_imputation_and_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def fill_age(row):
            if pd.notna(row["Age"]):
                return row["Age"]
            key = (row["Title"], row["Pclass"])
            return self.age_medians_.get(key, self.global_age_median_)

        df["Age"] = df.apply(fill_age, axis=1)
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)

        df["TicketGroupSize"] = df["Ticket"].map(self.ticket_counts_).fillna(1)
        df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

        keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["IsAlone"]
        df = df[keep]
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
        return df

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TitanicPreprocessor":
        return joblib.load(path)


def get_target(df: pd.DataFrame) -> np.ndarray:
    return df["Survived"].values.astype(np.float32)
