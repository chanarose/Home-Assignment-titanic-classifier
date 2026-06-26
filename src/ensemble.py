"""Save/load/predict helpers for an ensemble of TitanicMLP models.

Averaging several models trained with different random seeds on the same
data cancels out a lot of the per-seed noise that comes from random weight
init and minibatch order -- cheap variance reduction with no extra data.
"""

import json
import os

import numpy as np
import torch

from src.model import TitanicMLP


def save_ensemble(models, models_dir: str):
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(models_dir, f"titanic_model_{i}.pt"))


def load_ensemble(models_dir: str):
    with open(os.path.join(models_dir, "model_config.json")) as f:
        config = json.load(f)

    models = []
    for i in range(config["ensemble_size"]):
        path = os.path.join(models_dir, f"titanic_model_{i}.pt")
        model = TitanicMLP(
            config["input_dim"], tuple(config["hidden_dims"]), config["dropout"]
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models.append(model)
    return models


def predict_ensemble(models, X: torch.Tensor) -> np.ndarray:
    """Average the sigmoid probabilities of every ensemble member."""
    probs = []
    with torch.no_grad():
        for model in models:
            probs.append(torch.sigmoid(model(X)).numpy())
    return np.mean(probs, axis=0)
