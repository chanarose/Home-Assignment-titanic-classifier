"""Streamlit app for the Titanic survival classifier.

Two views:
  1. Training Results — shows the loss/accuracy curves and validation
     metrics produced by train.py.
  2. Run Inference — lets the user point at any Titanic-format CSV, loads
     the trained model + preprocessor from disk, runs predictions, and (if
     the CSV has a Survived column) reports evaluation metrics and plots.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.ensemble import load_ensemble, predict_ensemble
from src.preprocessing import TitanicPreprocessor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="Titanic Survival Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")

    if not os.path.exists(config_path) or not os.path.exists(preprocessor_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    weight_paths = [
        os.path.join(MODELS_DIR, f"titanic_model_{i}.pt")
        for i in range(config.get("ensemble_size", 1))
    ]
    if not all(os.path.exists(p) for p in weight_paths):
        return None

    members = load_ensemble(MODELS_DIR)
    preprocessor = TitanicPreprocessor.load(preprocessor_path)
    return members, preprocessor


def run_inference(members, preprocessor, df: pd.DataFrame):
    X = preprocessor.transform(df).values.astype(np.float32)
    probs = predict_ensemble(members, torch.tensor(X, dtype=torch.float32))
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Died", "Survived"])
    ax.set_yticklabels(["Died", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def plot_roc_curve(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig


st.title("Titanic Survival Classifier")

artifacts = load_artifacts()
if artifacts is None:
    st.error(
        "No trained model found in `models/`. Run `python train.py` first to "
        "produce the `titanic_model_*.pt` ensemble members, `preprocessor.joblib`, "
        "and `model_config.json`."
    )
    st.stop()

members, preprocessor = artifacts

tab_results, tab_inference = st.tabs(["Training Results", "Run Inference"])

with tab_results:
    st.header("Training history")
    history_path = os.path.join(MODELS_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.plot(history["train_loss"], label="Train loss")
            ax.plot(history["val_loss"], label="Validation loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("BCE Loss")
            ax.set_title("Loss curves")
            ax.legend()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(history["train_acc"], label="Train accuracy")
            ax.plot(history["val_acc"], label="Validation accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy curves")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("No history.json found — re-run train.py to generate training curves.")

    st.header("Validation set performance (held out by train.py)")
    metrics_path = os.path.join(MODELS_DIR, "val_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            val_metrics = json.load(f)

        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(key.replace("_", " ").title(), f"{val_metrics[key]:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            cm = np.array(val_metrics["confusion_matrix"])
            st.pyplot(plot_confusion_matrix(cm, title="Validation Confusion Matrix"))
        with col2:
            val_preds_path = os.path.join(MODELS_DIR, "val_predictions.csv")
            if os.path.exists(val_preds_path):
                val_preds_df = pd.read_csv(val_preds_path)
                st.pyplot(
                    plot_roc_curve(val_preds_df["Survived"], val_preds_df["PredictedProb"])
                )
    else:
        st.info("No val_metrics.json found — re-run train.py to generate validation metrics.")

    st.header("Cross-validated estimate (more robust than the single split above)")
    st.caption(
        "A single 80/20 split has only ~180 validation rows, so its accuracy can swing "
        "several points by chance depending on the random seed. The k-fold mean/std below "
        "averages over multiple splits for a more trustworthy estimate of true performance."
    )
    cv_path = os.path.join(MODELS_DIR, "cv_metrics.json")
    if os.path.exists(cv_path):
        with open(cv_path) as f:
            cv_metrics = json.load(f)

        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(
                key.replace("_", " ").title(),
                f"{cv_metrics[key]['mean']:.3f} ± {cv_metrics[key]['std']:.3f}",
            )
        st.caption(f"{cv_metrics['folds']}-fold stratified cross-validation over the full dataset.")
    else:
        st.info("No cv_metrics.json found — re-run train.py (with --cv-folds > 0) to generate it.")

with tab_inference:
    st.header("Run inference on a CSV")
    st.write(
        "Provide the path to a Titanic-format CSV (same columns as train.csv). "
        "If it includes a `Survived` column, evaluation metrics and plots will "
        "also be shown."
    )
    default_path = os.path.join(DATA_DIR, "val.csv")
    csv_path = st.text_input("Path to dataset CSV", value=default_path)
    labels_path = st.text_input(
        "Optional: path to a separate labels CSV with PassengerId + Survived "
        "(for datasets like Kaggle's test.csv that don't include the answer)",
        value="",
    )

    if st.button("Run inference"):
        if not os.path.exists(csv_path):
            st.error(f"File not found: {csv_path}")
        else:
            try:
                df = pd.read_csv(csv_path)
                probs, preds = run_inference(members, preprocessor, df)

                result_df = df.copy()
                result_df["PredictedProb"] = probs
                result_df["PredictedSurvived"] = preds

                st.subheader("Predictions")
                st.dataframe(result_df.head(50))
                st.download_button(
                    "Download predictions as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                y_true = None
                if "Survived" in df.columns:
                    y_true = df["Survived"].astype(int).values
                elif labels_path.strip():
                    if not os.path.exists(labels_path):
                        st.warning(f"Labels file not found: {labels_path} — showing predictions only.")
                    elif "PassengerId" not in df.columns:
                        st.warning(
                            "Dataset has no PassengerId column to join labels on — "
                            "showing predictions only."
                        )
                    else:
                        labels_df = pd.read_csv(labels_path)
                        joined = df[["PassengerId"]].merge(labels_df, on="PassengerId", how="left")
                        if joined["Survived"].isna().any():
                            st.warning(
                                "Some rows had no matching label in the labels CSV — "
                                "showing predictions only."
                            )
                        else:
                            y_true = joined["Survived"].astype(int).values
                            st.caption(
                                f"Evaluating against labels joined from `{labels_path}` "
                                "on PassengerId — not part of the training pipeline, "
                                "shown for inspection only."
                            )

                if y_true is not None:
                    metrics = {
                        "accuracy": accuracy_score(y_true, preds),
                        "precision": precision_score(y_true, preds, zero_division=0),
                        "recall": recall_score(y_true, preds, zero_division=0),
                        "f1": f1_score(y_true, preds, zero_division=0),
                        "roc_auc": roc_auc_score(y_true, probs),
                    }

                    st.subheader("Evaluation metrics")
                    cols = st.columns(5)
                    for col, (key, value) in zip(cols, metrics.items()):
                        col.metric(key.replace("_", " ").title(), f"{value:.3f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        cm = confusion_matrix(y_true, preds)
                        st.pyplot(plot_confusion_matrix(cm, title="Confusion Matrix"))
                    with col2:
                        st.pyplot(plot_roc_curve(y_true, probs))
                else:
                    st.info(
                        "No `Survived` column found in this CSV — showing predictions "
                        "only, no evaluation metrics."
                    )
            except Exception as exc:
                st.error(f"Failed to run inference: {exc}")
