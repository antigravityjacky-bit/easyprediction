"""
Ensemble Prediction — combines physics model + ML classifiers + LTR ranker.

No odds data used (pure physics + ML approach).

Default weights (tunable on validation set):
  - LightGBM classifier: 0.30
  - XGBoost classifier:  0.20
  - LambdaRank ranker:   0.30
  - Physics Monte Carlo:  0.20
"""

from __future__ import annotations

import numpy as np
from typing import Optional

DEFAULT_WEIGHTS = {
    "lgbm": 0.30,
    "xgb": 0.20,
    "ltr": 0.30,
    "physics": 0.20,
}


def ensemble_predict(
    lgbm_probs: np.ndarray | None = None,
    xgb_probs: np.ndarray | None = None,
    ltr_probs: np.ndarray | None = None,
    physics_probs: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Combine multiple model outputs into final top-3 probability per horse.

    All input arrays must have the same length (one entry per horse).
    Missing models have their weight redistributed proportionally.

    Parameters
    ----------
    lgbm_probs : LightGBM classifier P(top-3).
    xgb_probs : XGBoost classifier P(top-3).
    ltr_probs : LambdaRank softmax probabilities.
    physics_probs : Monte Carlo place probabilities from physics engine.
    weights : Custom weights dict. Keys: "lgbm", "xgb", "ltr", "physics".

    Returns
    -------
    np.ndarray : Combined probability per horse.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)

    # Collect available sources
    sources = {}
    if lgbm_probs is not None:
        sources["lgbm"] = lgbm_probs
    if xgb_probs is not None:
        sources["xgb"] = xgb_probs
    if ltr_probs is not None:
        sources["ltr"] = ltr_probs
    if physics_probs is not None:
        sources["physics"] = physics_probs

    if not sources:
        raise ValueError("At least one prediction source must be provided")

    # Redistribute weights for missing sources
    active_weight_sum = sum(w.get(k, 0) for k in sources)
    if active_weight_sum <= 0:
        active_weight_sum = 1.0

    normalized_weights = {
        k: w.get(k, 0) / active_weight_sum for k in sources
    }

    # Weighted average
    n = len(next(iter(sources.values())))
    combined = np.zeros(n)
    for key, probs in sources.items():
        combined += normalized_weights[key] * np.asarray(probs)

    return combined


def rank_horses(
    ensemble_probs: np.ndarray,
    horse_codes: list[str],
) -> list[tuple[str, float]]:
    """
    Sort horses by ensemble probability descending.

    Returns [(horse_code, probability), ...] sorted best-first.
    """
    pairs = list(zip(horse_codes, ensemble_probs.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def pick_top3(
    ensemble_probs: np.ndarray,
    horse_codes: list[str],
) -> list[str]:
    """Return top-3 horse codes by ensemble probability."""
    ranked = rank_horses(ensemble_probs, horse_codes)
    return [code for code, _ in ranked[:3]]


def tune_weights(
    predictions: list[dict],
    metric: str = "precision_at_3",
) -> dict[str, float]:
    """
    Grid-search optimal ensemble weights on validation data.

    Parameters
    ----------
    predictions : List of race-level dicts, each containing:
        - lgbm_probs, xgb_probs, ltr_probs, physics_probs (arrays)
        - horse_codes (list)
        - actual_top3 (set of horse codes that finished top-3)
    metric : Optimization metric ("precision_at_3" or "hit_rate").

    Returns
    -------
    dict : Best weights.
    """
    best_score = -1.0
    best_weights = dict(DEFAULT_WEIGHTS)

    # Coarse grid search over weight combinations
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for w_lgbm in grid:
        for w_xgb in grid:
            for w_ltr in grid:
                w_physics = 1.0 - w_lgbm - w_xgb - w_ltr
                if w_physics < -0.01 or w_physics > 0.51:
                    continue

                weights = {
                    "lgbm": w_lgbm, "xgb": w_xgb,
                    "ltr": w_ltr, "physics": max(0, w_physics),
                }

                score = _evaluate_weights(predictions, weights, metric)
                if score > best_score:
                    best_score = score
                    best_weights = weights

    return best_weights


def _evaluate_weights(
    predictions: list[dict],
    weights: dict[str, float],
    metric: str,
) -> float:
    """Evaluate ensemble weights on a set of race predictions."""
    hits = 0
    total = 0

    for race in predictions:
        combined = ensemble_predict(
            lgbm_probs=race.get("lgbm_probs"),
            xgb_probs=race.get("xgb_probs"),
            ltr_probs=race.get("ltr_probs"),
            physics_probs=race.get("physics_probs"),
            weights=weights,
        )
        codes = race["horse_codes"]
        actual_top3 = set(race["actual_top3"])

        predicted = pick_top3(combined, codes)

        if metric == "precision_at_3":
            hits += len(set(predicted) & actual_top3)
            total += 3
        elif metric == "hit_rate":
            hits += 1 if set(predicted) & actual_top3 else 0
            total += 1

    return hits / total if total > 0 else 0.0
