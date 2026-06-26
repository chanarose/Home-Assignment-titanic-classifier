"""Fetch and load the Titanic dataset.

Fetching uses the official Kaggle API so the pipeline is reproducible from
source. Kaggle requires authentication (a kaggle.json token under
~/.kaggle/ or the KAGGLE_USERNAME / KAGGLE_KEY env vars). If credentials are
not available and a train.csv already exists on disk, that local copy is
used instead so the rest of the pipeline still runs.
"""

import os
import zipfile

import pandas as pd

KAGGLE_COMPETITION = "titanic"


def fetch_titanic_data(data_dir: str) -> str:
    """Ensure data/train.csv exists locally, downloading it from Kaggle if needed.

    Returns the path to train.csv.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")

    if os.path.exists(train_path):
        return train_path

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_file(
            KAGGLE_COMPETITION, "train.csv", path=data_dir
        )
        zip_path = os.path.join(data_dir, "train.csv.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            os.remove(zip_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not download train.csv from Kaggle and no local copy was "
            f"found at {train_path}.\n"
            "Set up Kaggle API credentials (https://www.kaggle.com/settings -> "
            "'Create New Token', save kaggle.json to ~/.kaggle/) and accept the "
            "competition rules at "
            "https://www.kaggle.com/competitions/titanic, then re-run, or "
            f"manually place train.csv at {train_path}.\n"
            f"Original error: {exc}"
        ) from exc

    if not os.path.exists(train_path):
        raise RuntimeError(
            f"Download appeared to succeed but {train_path} is missing."
        )
    return train_path


def load_titanic_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
