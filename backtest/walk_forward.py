"""
Walk-Forward Backtest (v2) — honest ML evaluation without data leakage.

For each test window, trains models ONLY on data before that window.
Includes CatBoost and optimized ensemble for maximum accuracy.

Usage:
    python -m horseracing.backtest.walk_forward
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from horseracing.scraper.bulk_scraper import BULK_DIR
from horseracing.features.engineer import FEATURE_NAMES, VENUE_ENC, CONDITION_ENC, TREND_ENC, STYLE_ENC
from horseracing.ml.models import (
    train_lgbm_top3, train_xgb_top3, train_lgbm_ranker,
    predict_top3_prob, predict_rank_scores, rank_scores_to_probs,
)
from horseracing.ml.ensemble import ensemble_predict, pick_top3
from horseracing.ml.train import build_feature_matrix_from_dataset, _build_ltr_labels

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "datasets" / "processed" / "backtest"
PROCESSED_DIR = Path(__file__).resolve().parents[3] / "datasets" / "processed"


def _train_catboost(X_train, y_train, X_val=None, y_val=None):
    """Train CatBoost classifier."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3,
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=42,
    )
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    else:
        model.fit(X_train, y_train)
    return model


def _predict_catboost(model, X):
    probs = model.predict_proba(X)
    return probs[:, 1] if probs.ndim == 2 else probs


def _per_race_normalize(probs, groups):
    """Normalize probabilities to sum to 1 within each race group."""
    result = np.zeros_like(probs)
    idx = 0
    for g in groups:
        chunk = probs[idx:idx+g]
        s = chunk.sum()
        if s > 0:
            result[idx:idx+g] = chunk / s
        else:
            result[idx:idx+g] = 1.0 / g
        idx += g
    return result


def _smart_ensemble(model_probs_dict, groups, race_ids, y_true=None):
    """
    Smart ensemble: per-race softmax normalization + learned weights.
    If y_true is provided, tune weights via grid search on the validation set.
    """
    # Normalize each model's probs per-race
    normalized = {}
    for name, probs in model_probs_dict.items():
        if probs is not None:
            normalized[name] = _per_race_normalize(probs, groups)

    if not normalized:
        return np.ones(sum(groups)) / sum(groups)

    # Default equal weights
    n_models = len(normalized)
    names = list(normalized.keys())

    if y_true is not None and len(names) >= 2:
        # Grid search for best weights
        best_score = -1
        best_weights = {n: 1.0/n_models for n in names}

        # Generate weight combinations (step 0.1)
        from itertools import product
        steps = [round(x * 0.1, 1) for x in range(11)]
        for combo in product(steps, repeat=n_models):
            if abs(sum(combo) - 1.0) > 0.01:
                continue
            weights = dict(zip(names, combo))
            combined = sum(normalized[n] * weights[n] for n in names)

            # Evaluate
            score = _evaluate_ensemble(combined, groups, race_ids, y_true)
            if score > best_score:
                best_score = score
                best_weights = weights

        combined = sum(normalized[n] * best_weights[n] for n in names)
        return combined, best_weights
    else:
        # Equal weights
        combined = sum(normalized[n] / n_models for n in names)
        return combined, {n: 1.0/n_models for n in names}


def _evaluate_ensemble(combined, groups, race_ids, y_true):
    """Evaluate ensemble by hit rate (our primary target metric)."""
    idx = 0
    hits = 0
    total = 0
    unique_races = sorted(set(race_ids))
    race_starts = {}

    # Build race → index mapping
    curr_idx = 0
    for g in groups:
        if curr_idx < len(race_ids):
            rid = race_ids.iloc[curr_idx] if hasattr(race_ids, 'iloc') else race_ids[curr_idx]
            race_starts[rid] = (curr_idx, g)
        curr_idx += g

    for rid, (start, size) in race_starts.items():
        if size < 3:
            continue
        race_probs = combined[start:start+size]
        race_y = y_true.iloc[start:start+size] if hasattr(y_true, 'iloc') else y_true[start:start+size]

        top3_idx = np.argsort(race_probs)[-3:]
        actual_top3 = set(np.where(race_y == 1)[0])

        correct = len(set(top3_idx) & actual_top3)
        if correct > 0:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0


def _compute_physics_score(X, groups):
    """
    Composite physics score combining multiple signals.
    Each component is per-race ranked and normalized.
    """
    n = len(X)
    scores = np.zeros(n)

    # Component weights
    components = {
        "hist_asr_mean": 0.25,
        "hist_speed_mean": 0.15,
        "hist_place_rate": 0.20,
        "hist_avg_finish_pos": -0.15,  # Lower is better
        "hist_fap_mean": 0.10,
        "jockey_place_rate": 0.10,
        "last_asr": 0.05,
    }

    for col, weight in components.items():
        if col not in X.columns:
            continue
        vals = X[col].fillna(0).values.copy().astype(float)

        # Per-race rank normalization
        idx = 0
        ranked = np.zeros(n)
        for g in groups:
            chunk = vals[idx:idx+g]
            if weight < 0:
                chunk = -chunk  # Invert for "lower is better"
            # Min-max scale within race
            mn, mx = chunk.min(), chunk.max()
            if mx > mn:
                ranked[idx:idx+g] = (chunk - mn) / (mx - mn)
            else:
                ranked[idx:idx+g] = 0.5
            idx += g
        scores += ranked * abs(weight)

    # Normalize per race
    return _per_race_normalize(scores, groups)


def walk_forward_backtest(
    dataset_path: Path | None = None,
    n_windows: int = 5,
) -> pd.DataFrame:
    """
    Walk-forward evaluation with 5 models + smart ensemble.
    """
    if dataset_path is None:
        dataset_path = PROCESSED_DIR / "ml_dataset.csv"

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} rows, {df['race_id'].nunique()} races")

    df = df.sort_values(["date", "venue_code", "race_no", "finish_position"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    print(f"Date range: {dates[0]} to {dates[-1]} ({n_dates} race days)")

    chunk_size = max(1, n_dates // (n_windows + 1))
    all_results = []

    for win_i in range(n_windows):
        train_end = chunk_size * (win_i + 1)
        test_start = train_end
        test_end = min(train_end + chunk_size, n_dates)

        if test_start >= n_dates:
            break

        train_dates = set(dates[:train_end])
        test_dates = set(dates[test_start:test_end])

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        print(f"\n── Window {win_i + 1}/{n_windows} ─────────────")
        print(f"  Train: {len(train_df)} entries, {train_df['race_id'].nunique()} races")
        print(f"  Test:  {len(test_df)} entries, {test_df['race_id'].nunique()} races")

        X_train, y_train, groups_train = build_feature_matrix_from_dataset(train_df)
        X_test, y_test, groups_test = build_feature_matrix_from_dataset(test_df)
        y_ltr_train = _build_ltr_labels(train_df)

        # Train 4 models
        lgbm = train_lgbm_top3(X_train, y_train)
        xgb_model = train_xgb_top3(X_train, y_train)
        ltr = train_lgbm_ranker(X_train, y_ltr_train, groups_train)
        cat = _train_catboost(X_train, y_train)

        # Predict
        lgbm_probs = predict_top3_prob(lgbm, X_test)
        xgb_probs = predict_top3_prob(xgb_model, X_test)
        ltr_scores = predict_rank_scores(ltr, X_test)
        ltr_probs = rank_scores_to_probs(ltr_scores, groups_test)
        cat_probs = _predict_catboost(cat, X_test)

        # Enhanced physics score — combine multiple signals
        physics_probs = _compute_physics_score(X_test, groups_test)

        # Smart ensemble with weight tuning on a held-out portion of training data
        # Use last 20% of training as validation for weight tuning
        val_split = int(len(train_df) * 0.8)
        val_df = train_df.iloc[val_split:].copy()
        if len(val_df) > 30:
            X_val, y_val, groups_val = build_feature_matrix_from_dataset(val_df)
            y_ltr_val = _build_ltr_labels(val_df)

            val_lgbm = predict_top3_prob(lgbm, X_val)
            val_xgb = predict_top3_prob(xgb_model, X_val)
            val_ltr_scores = predict_rank_scores(ltr, X_val)
            val_ltr_probs = rank_scores_to_probs(val_ltr_scores, groups_val)
            val_cat = _predict_catboost(cat, X_val)
            val_physics = _compute_physics_score(X_val, groups_val)

            val_model_probs = {
                "lgbm": val_lgbm, "xgb": val_xgb, "ltr": val_ltr_probs,
                "cat": val_cat, "physics": val_physics,
            }
            _, best_weights = _smart_ensemble(
                val_model_probs, groups_val, val_df["race_id"], y_val
            )
        else:
            best_weights = {"lgbm": 0.2, "xgb": 0.15, "ltr": 0.3, "cat": 0.2, "physics": 0.15}

        print(f"  Weights: {best_weights}")

        # Apply ensemble to test
        test_model_probs = {
            "lgbm": lgbm_probs, "xgb": xgb_probs, "ltr": ltr_probs,
            "cat": cat_probs, "physics": physics_probs,
        }
        normalized = {n: _per_race_normalize(p, groups_test)
                      for n, p in test_model_probs.items() if p is not None}
        combined = sum(normalized[n] * best_weights.get(n, 0) for n in normalized)

        # Evaluate per race
        test_df["ensemble_prob"] = combined

        win_hits = 0
        win_hit2 = 0
        win_total = 0
        win_correct_sum = 0

        for race_id, race_df in test_df.groupby("race_id"):
            if len(race_df) < 3:
                continue
            actual_top3 = set(
                race_df[race_df["finish_position"] <= 3]["horse_code"].tolist()
            )
            if len(actual_top3) < 3:
                continue

            predicted = pick_top3(
                race_df["ensemble_prob"].values,
                race_df["horse_code"].tolist(),
            )
            correct = len(set(predicted) & actual_top3)

            all_results.append({
                "window": win_i + 1,
                "race_id": race_id,
                "date": race_df["date"].iloc[0],
                "correct_count": correct,
                "precision_at_3": correct / 3,
                "hit_any": 1 if correct > 0 else 0,
                "hit_2plus": 1 if correct >= 2 else 0,
            })

            if correct > 0:
                win_hits += 1
            if correct >= 2:
                win_hit2 += 1
            win_total += 1
            win_correct_sum += correct

        if win_total > 0:
            print(f"  Precision@3: {win_correct_sum / (win_total * 3):.3f}")
            print(f"  Hit Rate:    {win_hits / win_total:.3f}")
            print(f"  Hit ≥2:      {win_hit2 / win_total:.3f}")
            print(f"  Miss Rate:   {1 - win_hits / win_total:.3f}")

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        print(f"\n══ Walk-Forward Summary ══════════════════════════")
        print(f"  Total races: {len(results_df)}")
        print(f"  Avg Precision@3: {results_df['precision_at_3'].mean():.3f}")
        print(f"  Hit Rate (≥1):   {results_df['hit_any'].mean():.3f}")
        print(f"  Hit ���2 Rate:     {results_df['hit_2plus'].mean():.3f}")
        print(f"  Miss Rate (0/3): {(results_df['correct_count'] == 0).mean():.3f}")
        print(f"  Avg correct:     {results_df['correct_count'].mean():.2f}/3")

        # Per-window breakdown
        print(f"\n  Per-window breakdown:")
        for w in sorted(results_df["window"].unique()):
            wd = results_df[results_df["window"] == w]
            print(f"    Window {w}: P@3={wd['precision_at_3'].mean():.3f} "
                  f"Hit={wd['hit_any'].mean():.3f} "
                  f"Hit≥2={wd['hit_2plus'].mean():.3f} "
                  f"({len(wd)} races)")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(OUTPUT_DIR / "walk_forward_results.csv", index=False)

    return results_df


if __name__ == "__main__":
    walk_forward_backtest()
