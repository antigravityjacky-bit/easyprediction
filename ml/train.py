"""
Training Pipeline with Time-Series Cross-Validation

Trains LightGBM, XGBoost, and LambdaRank models using strict temporal
splits to prevent data leakage.

Usage (CLI):
    python -m horseracing.ml.train --dataset datasets/processed/ml_dataset.csv

Usage (Python):
    from horseracing.ml.train import train_and_evaluate
    results = train_and_evaluate()
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from horseracing.features.engineer import FEATURE_NAMES
from horseracing.ml.models import (
    train_lgbm_top3, train_xgb_top3, train_lgbm_ranker,
    predict_top3_prob, predict_rank_scores, rank_scores_to_probs,
    save_models,
)
from horseracing.ml.ensemble import (
    ensemble_predict, pick_top3, tune_weights,
)

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "datasets" / "processed"
MODEL_DIR = PROCESSED_DIR / "models"


# ── Time-Series CV Splits ────────────────────────────────────────────────────

def time_series_cv_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 4,
) -> list[tuple[pd.Index, pd.Index]]:
    """
    Generate expanding-window time-series cross-validation splits.

    Splits:
      Fold 1: Train Jan 1-31,    Val Feb 1-14
      Fold 2: Train Jan 1-Feb 14, Val Feb 15-28
      Fold 3: Train Jan 1-Feb 28, Val Mar 1-14
      Fold 4: Train Jan 1-Mar 14, Val Mar 15-26

    No shuffling. Strictly temporal ordering.
    """
    dates = sorted(df[date_col].unique())
    n_dates = len(dates)

    if n_dates < n_splits + 1:
        # Not enough dates for requested splits — use leave-last-out
        return [(
            df[df[date_col] < dates[-1]].index,
            df[df[date_col] == dates[-1]].index,
        )]

    # Divide dates into n_splits + 1 chunks
    chunk_size = n_dates // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end_idx = chunk_size * (i + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(train_end_idx + chunk_size, n_dates)

        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])

        train_idx = df[df[date_col].isin(train_dates)].index
        val_idx = df[df[date_col].isin(val_dates)].index

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


# ── Feature Matrix Builder ───────────────────────────────────────────────────

def build_feature_matrix_from_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """
    Build ML-ready feature matrix from the dataset CSV.

    Uses pre-computed columns where available, fills feature vector from
    the FEATURE_NAMES list. Applies encoding for categorical columns.

    Returns (X, y_top3, groups) where:
      X = DataFrame with FEATURE_NAMES columns
      y_top3 = binary series (1 if top-3, else 0)
      groups = list of field sizes per race (for LTR)
    """
    from horseracing.features.engineer import VENUE_ENC, CONDITION_ENC, TREND_ENC, STYLE_ENC

    # Add encoded columns if they don't exist
    if "venue_enc" not in df.columns and "venue" in df.columns:
        df = df.copy()
        df["venue_enc"] = df["venue"].map(VENUE_ENC).fillna(0)
    if "condition_enc" not in df.columns and "condition" in df.columns:
        df = df.copy()
        df["condition_enc"] = df["condition"].map(CONDITION_ENC).fillna(2)
    if "racing_style_enc" not in df.columns:
        if "racing_style" in df.columns:
            df = df.copy()
            df["racing_style_enc"] = df["racing_style"].map(STYLE_ENC).fillna(-1)
        else:
            df = df.copy()
            df["racing_style_enc"] = -1

    # Encode trend columns (string → int)
    for trend_col in ["hist_asr_trend", "hist_speed_trend", "hist_fap_trend"]:
        if trend_col in df.columns and df[trend_col].dtype == object:
            df = df.copy()
            df[trend_col] = df[trend_col].map(TREND_ENC).fillna(0)

    # Map existing columns to feature names
    feature_data = {}
    for fname in FEATURE_NAMES:
        if fname in df.columns:
            feature_data[fname] = pd.to_numeric(df[fname], errors="coerce").fillna(0).values
        else:
            feature_data[fname] = np.zeros(len(df))

    X = pd.DataFrame(feature_data, index=df.index)

    # Target
    y_top3 = (df["finish_position"].fillna(99).astype(int) <= 3).astype(int)

    # Groups (field sizes per race for LTR)
    groups = df.groupby("race_id").size().tolist()

    return X, y_top3, groups


def _build_ltr_labels(df: pd.DataFrame) -> pd.Series:
    """
    Build Learning-to-Rank relevance labels.
    Higher = better. Uses field_size - finish_position.
    """
    field_sizes = df.groupby("race_id")["finish_position"].transform("count")
    labels = field_sizes - df["finish_position"].fillna(field_sizes)
    return labels.clip(lower=0).astype(int)


# ── Training Pipeline ────────────────────────────────────────────────────────

def train_and_evaluate(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    n_cv_splits: int = 4,
) -> dict:
    """
    Full training pipeline:
      1. Load dataset
      2. Generate time-series CV splits
      3. Train LightGBM, XGBoost, LambdaRank on each fold
      4. Evaluate per-fold metrics
      5. Train final models on all data
      6. Save models

    Returns evaluation summary dict.
    """
    # Load dataset
    if dataset_path is None:
        dataset_path = PROCESSED_DIR / "ml_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} rows, {df['race_id'].nunique()} races")

    # Build feature matrix
    X, y_top3, groups = build_feature_matrix_from_dataset(df)
    y_ltr = _build_ltr_labels(df)

    # Time-series splits
    splits = time_series_cv_splits(df, date_col="date", n_splits=n_cv_splits)
    print(f"Generated {len(splits)} CV splits")

    # Per-fold evaluation
    fold_results = []
    all_val_predictions = []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n── Fold {fold_i + 1}/{len(splits)} ─────────────────────")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        X_train, X_val = X.loc[train_idx], X.loc[val_idx]
        y_train, y_val = y_top3.loc[train_idx], y_top3.loc[val_idx]

        # Train classifiers
        lgbm_model = train_lgbm_top3(X_train, y_train, X_val, y_val)
        xgb_model = train_xgb_top3(X_train, y_train, X_val, y_val)

        # Train ranker
        train_groups = df.loc[train_idx].groupby("race_id").size().tolist()
        val_groups = df.loc[val_idx].groupby("race_id").size().tolist()
        y_ltr_train = y_ltr.loc[train_idx]
        y_ltr_val = y_ltr.loc[val_idx]

        ltr_model = train_lgbm_ranker(
            X_train, y_ltr_train, train_groups,
            X_val, y_ltr_val, val_groups,
        )

        # Predict on validation
        lgbm_probs = predict_top3_prob(lgbm_model, X_val)
        xgb_probs = predict_top3_prob(xgb_model, X_val)
        ltr_scores = predict_rank_scores(ltr_model, X_val)
        ltr_probs = rank_scores_to_probs(ltr_scores, val_groups)

        # Ensemble
        combined = ensemble_predict(
            lgbm_probs=lgbm_probs,
            xgb_probs=xgb_probs,
            ltr_probs=ltr_probs,
        )

        # Evaluate per-race
        val_df = df.loc[val_idx].copy()
        val_df["lgbm_prob"] = lgbm_probs
        val_df["xgb_prob"] = xgb_probs
        val_df["ltr_prob"] = ltr_probs
        val_df["ensemble_prob"] = combined

        fold_metrics = _evaluate_predictions(val_df)
        fold_results.append(fold_metrics)
        print(f"  Precision@3: {fold_metrics['precision_at_3']:.3f}")
        print(f"  Hit rate (any 1 of 3): {fold_metrics['hit_rate']:.3f}")

        # Collect for weight tuning
        for race_id, race_df in val_df.groupby("race_id"):
            actual_top3 = set(
                race_df[race_df["finish_position"] <= 3]["horse_code"].tolist()
            )
            all_val_predictions.append({
                "lgbm_probs": race_df["lgbm_prob"].values,
                "xgb_probs": race_df["xgb_prob"].values,
                "ltr_probs": race_df["ltr_prob"].values,
                "horse_codes": race_df["horse_code"].tolist(),
                "actual_top3": list(actual_top3),
            })

    # Tune ensemble weights on all validation data
    print("\n── Tuning ensemble weights ──────────────────────────")
    best_weights = tune_weights(all_val_predictions, metric="precision_at_3")
    print(f"  Best weights: {best_weights}")

    # Train final models on all data
    print("\n── Training final models on all data ────────────────")
    all_groups = df.groupby("race_id").size().tolist()

    final_lgbm = train_lgbm_top3(X, y_top3)
    final_xgb = train_xgb_top3(X, y_top3)
    final_ltr = train_lgbm_ranker(X, y_ltr, all_groups)

    # Save models
    output_dir = output_dir or MODEL_DIR
    save_models(
        {"lgbm_top3": final_lgbm, "xgb_top3": final_xgb, "ltr_ranker": final_ltr},
        output_dir,
    )
    # Save weights
    weights_path = output_dir / "ensemble_weights.json"
    with open(weights_path, "w") as f:
        json.dump(best_weights, f, indent=2)
    print(f"  Models saved to: {output_dir}")

    # Summary
    avg_metrics = _average_fold_metrics(fold_results)
    print(f"\n── CV Summary ──────────────────────────────────────")
    print(f"  Avg Precision@3: {avg_metrics['precision_at_3']:.3f}")
    print(f"  Avg Hit Rate:    {avg_metrics['hit_rate']:.3f}")
    print(f"  Avg Recall@3:    {avg_metrics['recall_at_3']:.3f}")

    return {
        "fold_results": fold_results,
        "avg_metrics": avg_metrics,
        "best_weights": best_weights,
        "model_dir": str(output_dir),
        "n_samples": len(df),
        "n_races": df["race_id"].nunique(),
    }


# ── Evaluation helpers ───────────────────────────────────────────────────────

def _evaluate_predictions(val_df: pd.DataFrame) -> dict:
    """
    Evaluate predictions at the race level.

    Metrics:
    - precision_at_3: Of our top-3 picks, how many are actually top-3?
    - recall_at_3: Of actual top-3, how many did we pick?
    - hit_rate: Fraction of races where at least 1 of our top-3 is correct.
    """
    precision_scores = []
    recall_scores = []
    hits = 0
    total_races = 0

    for race_id, race_df in val_df.groupby("race_id"):
        if len(race_df) < 3:
            continue

        actual_top3 = set(
            race_df[race_df["finish_position"] <= 3]["horse_code"].tolist()
        )
        if not actual_top3:
            continue

        predicted = pick_top3(
            race_df["ensemble_prob"].values,
            race_df["horse_code"].tolist(),
        )

        correct = len(set(predicted) & actual_top3)
        precision_scores.append(correct / 3)
        recall_scores.append(correct / len(actual_top3))
        if correct > 0:
            hits += 1
        total_races += 1

    return {
        "precision_at_3": np.mean(precision_scores) if precision_scores else 0.0,
        "recall_at_3": np.mean(recall_scores) if recall_scores else 0.0,
        "hit_rate": hits / total_races if total_races > 0 else 0.0,
        "n_races": total_races,
    }


def _average_fold_metrics(fold_results: list[dict]) -> dict:
    """Average metrics across CV folds."""
    if not fold_results:
        return {"precision_at_3": 0, "recall_at_3": 0, "hit_rate": 0}
    keys = ["precision_at_3", "recall_at_3", "hit_rate"]
    return {k: np.mean([f[k] for f in fold_results]) for k in keys}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for horse racing prediction")
    parser.add_argument("--dataset", default=None, help="Path to ml_dataset.csv")
    parser.add_argument("--output", default=None, help="Model output directory")
    parser.add_argument("--folds", type=int, default=4, help="Number of CV folds")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    output_dir = Path(args.output) if args.output else None

    results = train_and_evaluate(
        dataset_path=dataset_path,
        output_dir=output_dir,
        n_cv_splits=args.folds,
    )

    print(f"\n── Done ────────────────────────────────────────────")
    print(f"  Total samples: {results['n_samples']}")
    print(f"  Total races:   {results['n_races']}")
    print(f"  Models saved:  {results['model_dir']}")


if __name__ == "__main__":
    main()
