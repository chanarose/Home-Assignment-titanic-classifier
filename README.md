# Titanic Survival Classifier

End-to-end classification pipeline on the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data): EDA, a PyTorch model trained by a standalone script, and a Streamlit app for evaluation + inference.

Only `train.csv` is used anywhere in this project. It is split internally into a train/validation set; `test.csv` and `gender_submission.csv` are never touched.

## Project structure

```
titanic-classifier/
├── data/
│   ├── sample_train.csv     # small sample (30 rows) for quick inspection / repo submission
│   └── val.csv               # validation split written by train.py (gitignored, regenerated)
├── models/                   # artifacts written by train.py (gitignored, regenerated)
│   ├── titanic_model_0.pt ... titanic_model_4.pt  # ensemble members
│   ├── preprocessor.joblib
│   ├── model_config.json
│   ├── history.json           # member 0's training curve
│   ├── cv_metrics.json        # 5-fold CV summary (the robust accuracy estimate)
│   ├── val_metrics.json       # ensemble-averaged metrics on the single 80/20 split
│   └── val_predictions.csv
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis
├── src/
│   ├── data_utils.py          # Kaggle fetch + CSV loading
│   ├── preprocessing.py       # feature engineering, imputation, encoding, scaling
│   ├── model.py                # PyTorch MLP definition
│   └── ensemble.py             # save/load/predict helpers for the model ensemble
├── train.py                    # standalone training script
├── ds_app.py                   # Streamlit app (results + inference)
└── requirements.txt
```

## Setup

```bash
git clone <your-repo-url>
cd titanic-classifier
pip install -r requirements.txt
```

### Kaggle API credentials

`train.py` and the EDA notebook fetch `train.csv` directly from Kaggle via the official API, so the pipeline is reproducible from source.

1. Create a Kaggle account and accept the competition rules at https://www.kaggle.com/competitions/titanic.
2. Go to https://www.kaggle.com/settings → API → "Create New Token". This downloads `kaggle.json`.
3. Save it to `~/.kaggle/kaggle.json` (or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables instead).

If no credentials are configured, the fetch step falls back to an existing `data/train.csv` if you already have one in place — otherwise it raises a clear error explaining how to authenticate.

## Run instructions

### 1. Train the model

```bash
python train.py
```

This will:
1. Fetch `train.csv` from Kaggle (or reuse `data/train.csv` if already present).
2. Split it into train (80%) / validation (20%), stratified by `Survived`.
3. Fit the preprocessing pipeline on the train split only, then apply it to both splits.
4. Train a PyTorch MLP, tracking train/validation loss and accuracy per epoch.
5. Save the best-validation-loss model weights, the fitted preprocessor, training history, and validation metrics/predictions to `models/`. The validation split itself is saved to `data/val.csv` so the Streamlit app has a ready-made labeled dataset to run inference on.

Useful flags:

```bash
python train.py --epochs 100 --lr 0.001 --hidden-dims 64 32 --batch-size 32
python train.py --data-path /path/to/local/train.csv   # skip the Kaggle fetch
```

### 2. Launch the Streamlit app

```bash
streamlit run ds_app.py
```

The app has two tabs:

- **Training Results** — loss/accuracy curves over training; validation metrics (accuracy, precision, recall, F1, ROC-AUC) with a confusion matrix and ROC curve from the single held-out 80/20 split; and a **cross-validated estimate** (mean ± std over 5 folds) that's far less noisy than any single split.
- **Run Inference** — enter the path to any Titanic-format CSV (defaults to `data/val.csv`). The app loads the trained model + preprocessor from `models/`, runs predictions, and displays/downloads them. If the CSV includes a `Survived` column, it also reports evaluation metrics and plots for that data.

## Architecture and design choices

**Preprocessing (`src/preprocessing.py`)** — fit once on the training split and reused identically for validation and inference, so there is no train/inference skew:
- `Title` extracted from `Name` via regex (Mr/Mrs/Miss/Master/Rare), a strong low-cardinality proxy for age/sex/social status.
- `FamilySize = SibSp + Parch + 1` and `IsAlone`, since family size has a non-monotonic relationship with survival.
- `Deck` derived from the first letter of `Cabin` (`"U"` for unknown) rather than dropping the mostly-missing `Cabin` column outright.
- `TicketGroupSize` — count of passengers sharing the same `Ticket` number (fit on the training fold only, unseen tickets default to 1). Catches travel groups that `FamilySize` misses (friends, servants, nannies on a shared ticket).
- `FarePerPerson = Fare / TicketGroupSize` — `Fare` in the raw data is the *total* paid for the ticket, often shared across a group, so dividing it out gives a more honest per-passenger price than raw `Fare`.
- `SexPclass` — explicit interaction category (e.g. `female_1`) so the model doesn't have to reconstruct the sex×class effect from two separate one-hot blocks.
- `Age` imputed with the median per (`Title`, `Pclass`) group; `Fare` with the global median; `Embarked` with the mode.
- Categorical features (`Pclass`, `Sex`, `Embarked`, `Title`, `Deck`, `SexPclass`) one-hot encoded; numeric features standardized (`StandardScaler`), since the model is scale-sensitive.
- The fitted preprocessor (medians, ticket-group counts, encoding columns, scaler) is serialized with `joblib` and reloaded by the Streamlit app for consistent inference.

**Model (`src/model.py`)** — a small feed-forward network (`TitanicMLP`): configurable hidden layers (default 64→32) with ReLU + dropout, single logit output, trained with `BCEWithLogitsLoss` (using a `pos_weight` to account for the mild class imbalance) and Adam. The best-validation-loss checkpoint is kept rather than the final epoch.

**Training script (`train.py`)** — fetch → cross-validate → split → fit preprocessing → train → evaluate → persist everything the app needs (weights, preprocessor, config, history, metrics, CV summary). Fully parameterized via CLI flags.

- **Cross-validation before the final fit**: with only ~179 validation rows in a single 80/20 split, accuracy estimates are noisy — a sweep across 5 random seeds on this dataset showed swings of several points purely from which rows landed in validation. `train.py` runs 5-fold stratified CV over the *whole* dataset first (refitting the preprocessor per fold, so no leakage) and reports mean ± std accuracy/F1/ROC-AUC to `models/cv_metrics.json` — that's the number to trust, not any single split's accuracy. Disable with `--cv-folds 0`.
- **Default seed (`--seed`, default `1`)**: chosen from a 1-5 seed sweep as a representative, non-extreme split — not cherry-picked for the highest accidental accuracy. The single-split metrics in `models/val_metrics.json` will still vary a bit with `--seed`; the CV summary is the stable reference point.
- **Model ensemble (`--ensemble-size`, default `5`)**: the final deployed model is 5 `TitanicMLP`s trained on the *same* train/val split but different random seeds (init + minibatch order), with predictions averaged (`src/ensemble.py`). This doesn't add information the model didn't already have, but it cancels out per-seed noise — the 5 members landed within 0.6pp of each other (0.838-0.844 accuracy) versus the multi-point swings seen from single models trained on different splits. Set `--ensemble-size 1` for a single model.

**Streamlit app (`ds_app.py`)** — separates "what happened during training" (Training Results tab, sourced from artifacts written by `train.py`) from "run this model on arbitrary data" (Run Inference tab, which re-derives metrics live from whatever CSV is provided). Both reuse the same `TitanicPreprocessor` and `TitanicMLP` so results are guaranteed consistent with training.

## Example usage

```bash
python train.py
# Running 5-fold stratified cross-validation...
#   Fold 1/5: accuracy=0.7877 f1=0.7324 roc_auc=0.8522
#   ...
# CV accuracy: 0.8295 +/- 0.0264 (roc_auc: 0.8862 +/- 0.0240)
#
# Final fit -- Train rows: 712 (79.9%), Validation rows: 179 (20.1%)
#
# Training an ensemble of 5 member(s)...
#   Member 0: val_accuracy=0.8380
#   Member 1: val_accuracy=0.8436
#   Member 2: val_accuracy=0.8436
#   Member 3: val_accuracy=0.8436
#   Member 4: val_accuracy=0.8380
#
# Ensemble validation metrics (averaged over 5 members, single 80/20 split):
#   accuracy: 0.8436
#   precision: 0.7971
#   recall: 0.7971
#   f1: 0.7971
#   roc_auc: 0.8758

streamlit run ds_app.py
# Open http://localhost:8501
# Tab "Training Results": loss/accuracy curves, single-split confusion matrix + ROC curve,
#   and the 5-fold CV mean ± std (the number to trust over any one split)
# Tab "Run Inference": point at data/val.csv (or any other Titanic-format CSV) and click "Run inference"
```

The CV accuracy (0.8295 ± 0.026) is the reliable estimate of model performance; the single-split ensemble number (0.844 here) will still vary a bit depending on `--seed` purely from which ~179 rows land in validation — that's expected sampling noise on a dataset this small, not a flaw in the model. The ensemble's value shows up in how tight its 5 members are (0.838-0.844) compared to single-model runs across different seeds (0.80-0.87).

## Reproducibility notes

- Random seed fixed (`--seed`, default `1`) for the train/validation split; chosen as a representative seed from a small sweep, not cherry-picked for the best score.
- Final model is a 5-member ensemble (`--ensemble-size`, default `5`) of identical `TitanicMLP`s trained with different seeds on the same split, predictions averaged at inference.
- 5-fold stratified cross-validation (`--cv-folds`, default `5`) runs before the final fit and writes `models/cv_metrics.json`, giving a stable accuracy estimate independent of the single split's seed sensitivity.
- `data/train.csv` and everything under `models/` are regenerated by `train.py` and are gitignored — only `src/`, `train.py`, `ds_app.py`, the notebook, and a small `data/sample_train.csv` sample are committed.
