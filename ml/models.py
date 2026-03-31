"""
ML Models for Horse Racing Top-3 Prediction

Three model types:
  1. LightGBM binary classifier (top-3 vs not)
  2. XGBoost binary classifier (ensemble diversity)
  3. LightGBM LambdaRank (learning-to-rank within race)

All models handle NaN features natively (LightGBM/XGBoost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

import lightgbm as lgb
import xgboost as xgb


# ── Default hyperparameters ──────────────────────────────────────────────────
# Tuned for small datasets (~2000-4000 samples) with ~50 features.

LGBM_CLASSIFIER_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_samples": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "max_depth": 8,
    "verbose": -1,
    "random_state": 42,
    "is_unbalance": True,
    "n_jobs": 1,
}

XGB_CLASSIFIER_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "scale_pos_weight": 3.0,
    "verbosity": 0,
    "random_state": 42,
}

LGBM_RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_samples": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "max_depth": 8,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": 1,
}


# ── Training functions ───────────────────────────────────────────────────────

def train_lgbm_top3(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    """
    Train LightGBM binary classifier for top-3 prediction.

    Parameters
    ----------
    X_train : DataFrame with feature columns.
    y_train : Series with 1 (top-3) / 0 (not top-3).
    X_val, y_val : Optional validation set for early stopping.
    params : Override default hyperparameters.

    Returns trained LGBMClassifier.
    """
    p = {**LGBM_CLASSIFIER_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators", 300)

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **p)

    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False)]

    model.fit(X_train, y_train, **fit_params)
    return model


def train_xgb_top3(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """
    Train XGBoost binary classifier for top-3 prediction.
    """
    p = {**XGB_CLASSIFIER_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators", 300)

    model = xgb.XGBClassifier(n_estimators=n_estimators, **p)

    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val.values, y_val.values)]
        fit_params["verbose"] = False

    model.fit(X_train.values, y_train.values, **fit_params)
    return model


def train_lgbm_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: list[int],
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    groups_val: list[int] | None = None,
    params: dict | None = None,
) -> lgb.LGBMRanker:
    """
    Train LightGBM LambdaRank model for within-race ranking.

    Parameters
    ----------
    y_train : Relevance labels. Use (field_size - finish_position) so
              1st place gets highest label.
    groups_train : Number of horses per race (for grouping).
    """
    p = {**LGBM_RANKER_PARAMS, **(params or {})}
    n_estimators = p.pop("n_estimators", 300)

    model = lgb.LGBMRanker(n_estimators=n_estimators, **p)

    fit_params = {"group": groups_train}
    if X_val is not None and y_val is not None and groups_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["eval_group"] = [groups_val]
        fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False)]

    model.fit(X_train, y_train, **fit_params)
    return model


# ── Prediction functions ─────────────────────────────────────────────────────

def predict_top3_prob(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Predict top-3 probability for each horse.

    Works with LGBMClassifier and XGBClassifier.
    Returns 1D array of probabilities.
    """
    if isinstance(model, xgb.XGBClassifier):
        return model.predict_proba(X.values)[:, 1]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 else probs
    return model.predict(X)


def predict_rank_scores(model: lgb.LGBMRanker, X: pd.DataFrame) -> np.ndarray:
    """
    Predict ranking scores. Higher = more likely to finish first.
    Returns 1D array of raw scores (not probabilities).
    """
    return model.predict(X)


def rank_scores_to_probs(scores: np.ndarray, groups: list[int]) -> np.ndarray:
    """
    Convert raw ranking scores to within-race probabilities via softmax.

    Parameters
    ----------
    scores : Raw LambdaRank scores.
    groups : Number of horses per race (same order as scores).

    Returns 1D array of probabilities (sum to 1 within each race group).
    """
    probs = np.zeros_like(scores)
    idx = 0
    for g in groups:
        race_scores = scores[idx:idx + g]
        # Softmax with numerical stability
        shifted = race_scores - np.max(race_scores)
        exp_scores = np.exp(shifted)
        probs[idx:idx + g] = exp_scores / exp_scores.sum()
        idx += g
    return probs


# ── Model persistence ────────────────────────────────────────────────────────

def save_models(
    models: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save all models to disk."""
    import joblib
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}.pkl")


def load_models(model_dir: Path) -> dict[str, Any]:
    """Load all models from disk."""
    import joblib
    models = {}
    for path in model_dir.glob("*.pkl"):
        models[path.stem] = joblib.load(path)
    return models
