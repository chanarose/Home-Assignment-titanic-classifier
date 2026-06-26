"""Standalone training script for the Titanic survival classifier.

Loads train.csv (fetching it from Kaggle if missing), preprocesses it,
trains a PyTorch MLP, and saves the trained weights plus all artifacts the
Streamlit app needs to reproduce evaluation and run inference.

Before the final fit, runs stratified k-fold cross-validation over the
whole dataset to report a noise-robust accuracy estimate -- a single
80/20 split of ~180 validation rows has enough sampling variance that its
accuracy alone can swing several points seed-to-seed.

Usage:
    python train.py
    python train.py --epochs 100 --lr 0.001 --hidden-dims 64 32
    python train.py --cv-folds 0   # skip cross-validation
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data_utils import fetch_titanic_data, load_titanic_csv
from src.ensemble import save_ensemble
from src.model import TitanicMLP
from src.preprocessing import TitanicPreprocessor, get_target

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")


class TitanicDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Titanic survival classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for the train/val split and model init. Chosen from a "
        "5-seed sweep (1-5) as a representative, non-extreme split -- see README.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds to run for a robust accuracy estimate "
        "before the final fit. Set to 0 to skip.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of TitanicMLP members (different seeds, same data) to train and "
        "average for the final deployed model. Reduces variance from random init/"
        "minibatch order. Set to 1 for a single model.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to a local train.csv. If omitted, fetches via Kaggle API "
        "or falls back to data/train.csv if present.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    y_np = y.numpy().astype(int)
    metrics = {
        "accuracy": accuracy_score(y_np, preds),
        "precision": precision_score(y_np, preds, zero_division=0),
        "recall": recall_score(y_np, preds, zero_division=0),
        "f1": f1_score(y_np, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_np, probs),
        "confusion_matrix": confusion_matrix(y_np, preds).tolist(),
    }
    return metrics, probs, preds


def train_model(X_train, y_train, X_val, y_val, args, track_history=False):
    """Train a fresh TitanicMLP on one train/val split. Returns the best-val-loss
    model plus its validation metrics (and per-epoch history if requested)."""
    input_dim = X_train.shape[1]
    model = TitanicMLP(input_dim, hidden_dims=tuple(args.hidden_dims), dropout=args.dropout)

    pos_weight = torch.tensor((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TitanicDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        train_loss = epoch_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            train_logits = model(X_train_t)
            train_acc = ((torch.sigmoid(train_logits) >= 0.5).float() == y_train_t).float().mean().item()
            val_acc = ((torch.sigmoid(val_logits) >= 0.5).float() == y_val_t).float().mean().item()

        if track_history:
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if track_history and (epoch == 1 or epoch % 10 == 0 or epoch == args.epochs):
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    val_metrics, val_probs, val_preds = evaluate(model, X_val_t, y_val_t)
    return model, history, val_metrics, val_probs, val_preds


def cross_validate(df, args):
    """Stratified k-fold CV over the whole dataset for a robust accuracy estimate.

    Each fold fits its own TitanicPreprocessor on that fold's training rows only,
    so there's no leakage between folds.
    """
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Survived"]), start=1):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]

        preprocessor = TitanicPreprocessor()
        X_train = preprocessor.fit_transform(train_fold).values.astype(np.float32)
        y_train = get_target(train_fold)
        X_val = preprocessor.transform(val_fold).values.astype(np.float32)
        y_val = get_target(val_fold)

        _, _, val_metrics, _, _ = train_model(X_train, y_train, X_val, y_val, args)
        fold_metrics.append(val_metrics)
        print(
            f"  Fold {fold}/{args.cv_folds}: accuracy={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f} roc_auc={val_metrics['roc_auc']:.4f}"
        )

    summary = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        values = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    summary["folds"] = args.cv_folds
    summary["fold_results"] = fold_metrics
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    data_path = args.data_path or fetch_titanic_data(DATA_DIR)
    print(f"Loading data from {data_path}")
    df = load_titanic_csv(data_path)

    if args.cv_folds > 0:
        print(f"\nRunning {args.cv_folds}-fold stratified cross-validation...")
        cv_summary = cross_validate(df, args)
        print(
            f"\nCV accuracy: {cv_summary['accuracy']['mean']:.4f} "
            f"+/- {cv_summary['accuracy']['std']:.4f} "
            f"(roc_auc: {cv_summary['roc_auc']['mean']:.4f} +/- {cv_summary['roc_auc']['std']:.4f})"
        )
        with open(os.path.join(MODELS_DIR, "cv_metrics.json"), "w") as f:
            json.dump(cv_summary, f, indent=2)

    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["Survived"]
    )
    print(
        f"\nFinal fit -- Train rows: {len(train_df)} ({len(train_df) / len(df):.1%}), "
        f"Validation rows: {len(val_df)} ({len(val_df) / len(df):.1%})"
    )

    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)

    preprocessor = TitanicPreprocessor()
    X_train = preprocessor.fit_transform(train_df).values.astype(np.float32)
    y_train = get_target(train_df)
    X_val = preprocessor.transform(val_df).values.astype(np.float32)
    y_val = get_target(val_df)
    preprocessor.save(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    print(f"\nTraining an ensemble of {args.ensemble_size} member(s)...")
    members = []
    member_val_probs = []
    history = None
    for m in range(args.ensemble_size):
        set_seed(args.seed * 1000 + m)
        model, member_history, member_metrics, member_probs, _ = train_model(
            X_train, y_train, X_val, y_val, args, track_history=(m == 0)
        )
        members.append(model)
        member_val_probs.append(member_probs)
        if m == 0:
            history = member_history
        print(f"  Member {m}: val_accuracy={member_metrics['accuracy']:.4f}")

    old_single_model_path = os.path.join(MODELS_DIR, "titanic_model.pt")
    if os.path.exists(old_single_model_path):
        os.remove(old_single_model_path)
    save_ensemble(members, MODELS_DIR)

    val_probs = np.mean(member_val_probs, axis=0)
    val_preds = (val_probs >= 0.5).astype(int)
    val_metrics = {
        "accuracy": accuracy_score(y_val.astype(int), val_preds),
        "precision": precision_score(y_val.astype(int), val_preds, zero_division=0),
        "recall": recall_score(y_val.astype(int), val_preds, zero_division=0),
        "f1": f1_score(y_val.astype(int), val_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_val.astype(int), val_probs),
        "confusion_matrix": confusion_matrix(y_val.astype(int), val_preds).tolist(),
    }

    model_config = {
        "input_dim": X_train.shape[1],
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "ensemble_size": args.ensemble_size,
    }
    with open(os.path.join(MODELS_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(MODELS_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(MODELS_DIR, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    val_df_out = val_df.copy()
    val_df_out["PredictedProb"] = val_probs
    val_df_out["PredictedSurvived"] = val_preds
    val_df_out.to_csv(os.path.join(MODELS_DIR, "val_predictions.csv"), index=False)

    print(f"\nEnsemble validation metrics (averaged over {args.ensemble_size} members, single 80/20 split):")
    for k, v in val_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.4f}")
    print(f"  confusion_matrix: {val_metrics['confusion_matrix']}")
    print(f"\nSaved {args.ensemble_size} model member(s), preprocessor, and metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
