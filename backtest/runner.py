"""
Backtest Runner (v2) — evaluate predictions against actual results.

Uses the same rolling-history approach as dataset.py for consistency.
Processes races chronologically, building history incrementally.

Usage (CLI):
    python -m horseracing.backtest.runner --start 2026/01/01 --end 2026/03/26
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from horseracing.scraper.bulk_scraper import BULK_DIR
from horseracing.features.engineer import FEATURE_NAMES
from horseracing.ml.models import (
    load_models, predict_top3_prob, predict_rank_scores,
    rank_scores_to_probs,
)
from horseracing.ml.ensemble import ensemble_predict, pick_top3
from horseracing.ml.dataset import RollingTracker, _entry_to_race_entry, _parse_date, _days_between, _safe_float, _race_id
from horseracing.profile.builder import compute_metrics, build_profile_from_entries
from horseracing.metrics.advanced import compute_all_advanced
from horseracing.features.engineer import VENUE_ENC, CONDITION_ENC, TREND_ENC, STYLE_ENC

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "datasets" / "processed" / "backtest"


def _load_all_races(bulk_dir: Path) -> list[dict]:
    races = []
    for path in sorted(bulk_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                race = json.load(f)
            if race.get("entries"):
                races.append(race)
        except Exception:
            continue
    races.sort(key=lambda r: (r.get("date", ""), r.get("venue_code", ""), r.get("race_no", 0)))
    return races


def _build_features_from_tracker(entry: dict, race: dict, tracker: RollingTracker) -> dict:
    """
    Build the same feature vector as dataset.py, using the rolling tracker
    for history. This ensures backtest features match training features exactly.
    """
    horse_code = entry.get("horse_code", "")
    jockey = entry.get("jockey", "")
    venue = race.get("venue", "SHA_TIN")
    distance = race.get("distance", 0) or 0
    race_date = _parse_date(race.get("date", ""))
    field_size = len(race.get("entries", []))

    features = {
        "distance": distance,
        "venue_enc": VENUE_ENC.get(venue, 0),
        "condition_enc": CONDITION_ENC.get(race.get("condition", "GOOD"), 2),
        "field_size": field_size,
        "gate": entry.get("gate", 0) or 0,
        "draw_weight_lb": _safe_float(entry.get("draw_weight_lb")),
        "horse_weight_kg": _safe_float(entry.get("horse_weight_kg")),
    }

    # Field average weight
    field_weights = [_safe_float(e.get("draw_weight_lb")) for e in race.get("entries", [])
                     if _safe_float(e.get("draw_weight_lb")) > 0]
    avg_field_weight = sum(field_weights) / len(field_weights) if field_weights else 0
    features["weight_vs_field_avg"] = features["draw_weight_lb"] - avg_field_weight

    hist_entries = tracker.get_horse_entries(horse_code)
    hist_raw = tracker.get_horse_raw(horse_code)
    n_hist = len(hist_entries)
    features["n_history_races"] = n_hist

    if n_hist > 0:
        features["days_since_last"] = _days_between(hist_entries[-1].date, race_date)

        # Horse performance stats
        hist_fps = [e.finish_position for e in hist_entries if e.finish_position > 0]
        if hist_fps:
            features["hist_avg_finish_pos"] = sum(hist_fps) / len(hist_fps)
            features["hist_best_finish_pos"] = min(hist_fps)
            features["hist_win_rate"] = sum(1 for fp in hist_fps if fp == 1) / len(hist_fps)
            features["hist_place_rate"] = sum(1 for fp in hist_fps if fp <= 3) / len(hist_fps)
        else:
            features["hist_avg_finish_pos"] = 7
            features["hist_best_finish_pos"] = 14
            features["hist_win_rate"] = 0
            features["hist_place_rate"] = 0

        # Profile from last 6 races
        try:
            profile = build_profile_from_entries(horse_code, horse_code, hist_entries[-6:])
            if profile.asr:
                features["hist_asr_mean"] = profile.asr.mean
                features["hist_asr_std"] = profile.asr.std
                features["hist_asr_trend"] = TREND_ENC.get(profile.asr.trend, 0)
            if profile.true_speed:
                features["hist_speed_mean"] = profile.true_speed.mean
                features["hist_speed_std"] = profile.true_speed.std
                features["hist_speed_trend"] = TREND_ENC.get(profile.true_speed.trend, 0)
            if profile.fap:
                features["hist_fap_mean"] = profile.fap.mean
                features["hist_fap_std"] = profile.fap.std
                features["hist_fap_trend"] = TREND_ENC.get(profile.fap.trend, 0)
            if profile.fi:
                features["hist_fi_mean"] = profile.fi.mean
                features["hist_fi_std"] = profile.fi.std
            features["hist_edi_mean"] = _safe_float(profile.edi)
            features["hist_pa_mean"] = _safe_float(profile.pa)
            features["racing_style_enc"] = STYLE_ENC.get(profile.racing_style or "unknown", -1)
        except Exception:
            pass

        # Advanced metrics from most recent race
        latest = hist_raw[-1]
        sect = latest.get("section_times", []) or []
        try:
            from horseracing.metrics.asr import calculate_asr
            recent_asrs = []
            for hr in hist_raw[-6:]:
                ft = _safe_float(hr.get("finish_time"))
                d = hr.get("distance", 0) or 0
                w = _safe_float(hr.get("draw_weight_lb"))
                c = hr.get("condition", "GOOD")
                if ft > 0 and d > 0 and w > 0:
                    try:
                        recent_asrs.append(calculate_asr(d, ft, w, c))
                    except Exception:
                        pass

            adv = compute_all_advanced(
                section_times=sect,
                position_calls=latest.get("position_calls", []) or [],
                finish_time=_safe_float(latest.get("finish_time")),
                distance=latest.get("distance", 0) or 0,
                venue=latest.get("venue", venue),
                rail=latest.get("rail", "A"),
                gate=latest.get("gate", 0) or 0,
                horse_weight_kg=latest.get("horse_weight_kg", 0) or 0,
                carried_weight_lb=_safe_float(latest.get("draw_weight_lb")),
                days_since_last=features.get("days_since_last", -1),
                recent_asr_values=recent_asrs,
                history=hist_raw,
            )
            for key in ["peak_speed_ms", "finishing_burst", "speed_decay_rate",
                        "max_acceleration", "kinetic_energy_index", "weight_efficiency",
                        "power_output_watts", "turn_penalty_ms", "draw_extra_distance_m",
                        "positioning_cost", "drafting_factor", "form_trend",
                        "freshness", "distance_aptitude_ms", "track_affinity_ms"]:
                features[key] = _safe_float(adv.get(key))
        except Exception:
            pass

        # Venue and distance affinity
        venue_races = [e for e in hist_entries if e.venue == venue]
        features["n_races_same_venue"] = len(venue_races)
        if venue_races:
            venue_fps = [e.finish_position for e in venue_races if e.finish_position > 0]
            features["venue_avg_finish"] = sum(venue_fps) / len(venue_fps) if venue_fps else 7
            features["venue_place_rate"] = sum(1 for fp in venue_fps if fp <= 3) / len(venue_fps) if venue_fps else 0

        dist_races = [e for e in hist_entries if abs(e.distance - distance) <= 200]
        features["n_races_similar_dist"] = len(dist_races)
        if dist_races:
            dist_fps = [e.finish_position for e in dist_races if e.finish_position > 0]
            features["dist_avg_finish"] = sum(dist_fps) / len(dist_fps) if dist_fps else 7
            features["dist_place_rate"] = sum(1 for fp in dist_fps if fp <= 3) / len(dist_fps) if dist_fps else 0

        # Last race
        features["last_finish_pos"] = hist_entries[-1].finish_position
        try:
            last_m = compute_metrics(hist_entries[-1])
            features["last_asr"] = last_m.asr
            features["last_speed_ms"] = last_m.true_speed_ms
            features["last_fap"] = last_m.fap
        except Exception:
            pass

        features["dist_change"] = distance - (hist_entries[-1].distance or 0)
        features["weight_change"] = _safe_float(entry.get("draw_weight_lb")) - hist_entries[-1].draw_weight_lb
    else:
        features["days_since_last"] = -1
        features["hist_avg_finish_pos"] = 7
        features["hist_best_finish_pos"] = 14
        features["hist_win_rate"] = 0
        features["hist_place_rate"] = 0
        features["n_races_same_venue"] = 0
        features["n_races_similar_dist"] = 0
        features["last_finish_pos"] = 0
        features["last_asr"] = 0
        features["dist_change"] = 0
        features["weight_change"] = 0

    # Section time metadata
    sect_times = entry.get("section_times", []) or []
    features["has_section_times"] = 1 if len(sect_times) >= 2 else 0
    features["n_sections"] = len(sect_times)

    # Jockey stats
    jstats = tracker.get_jockey_stats(jockey)
    features.update(jstats)

    # Interactions
    features["gate_x_venue"] = features["gate"] * features["venue_enc"]
    features["weight_x_distance"] = features["draw_weight_lb"] * distance / 1000.0

    return features


def run_backtest(
    start_date: str = "2026/01/01",
    end_date: str = "2026/03/26",
    model_dir: Path | None = None,
    bulk_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    bulk_dir = bulk_dir or BULK_DIR
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_races = _load_all_races(bulk_dir)
    print(f"Loaded {len(all_races)} races from {bulk_dir}")

    races_in_range = [
        r for r in all_races
        if start_date <= r.get("date", "") <= end_date
    ]
    print(f"Races in range {start_date}-{end_date}: {len(races_in_range)}")

    if not races_in_range:
        return pd.DataFrame()

    # Load ML models
    ml_models = {}
    if model_dir and model_dir.exists():
        try:
            ml_models = load_models(model_dir)
            print(f"Loaded ML models: {list(ml_models.keys())}")
        except Exception as e:
            print(f"Warning: could not load ML models: {e}")

    ensemble_weights = None
    if model_dir:
        weights_path = model_dir / "ensemble_weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                ensemble_weights = json.load(f)

    # Process ALL races chronologically (for building history)
    tracker = RollingTracker()
    results = []
    horse_predictions = []

    for race in all_races:
        race_date = _parse_date(race.get("date", ""))
        rid = _race_id(race_date, race.get("venue_code", "ST"), race.get("race_no", 0))
        entries = race.get("entries", [])
        field_size = len(entries)
        in_range = start_date <= race.get("date", "") <= end_date

        if in_range and field_size >= 3:
            # Run prediction on this race
            venue = race.get("venue", "SHA_TIN")
            distance = race.get("distance", 0)

            # Actual top-3
            valid = [e for e in entries if e.get("finish_position", 0) and e.get("horse_code")]
            sorted_entries = sorted(valid, key=lambda e: e["finish_position"])
            actual_top3 = [e["horse_code"] for e in sorted_entries[:3]]
            actual_top3_set = set(actual_top3)

            # Shuffle to avoid order bias
            shuffled = list(entries)
            random.Random(hash(rid)).shuffle(shuffled)

            horse_codes = []
            feature_rows = []

            for entry in shuffled:
                code = entry.get("horse_code", "").strip()
                if not code:
                    continue
                features = _build_features_from_tracker(entry, race, tracker)
                horse_codes.append(code)
                feature_rows.append(features)

            if len(horse_codes) >= 3:
                X = pd.DataFrame(feature_rows)
                for col in FEATURE_NAMES:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0)

                # ML predictions
                lgbm_probs = xgb_probs = ltr_probs = None
                if "lgbm_top3" in ml_models:
                    try:
                        lgbm_probs = predict_top3_prob(ml_models["lgbm_top3"], X)
                    except Exception:
                        pass
                if "xgb_top3" in ml_models:
                    try:
                        xgb_probs = predict_top3_prob(ml_models["xgb_top3"], X)
                    except Exception:
                        pass
                if "ltr_ranker" in ml_models:
                    try:
                        ltr_scores = predict_rank_scores(ml_models["ltr_ranker"], X)
                        ltr_probs = rank_scores_to_probs(ltr_scores, [len(horse_codes)])
                    except Exception:
                        pass

                # Physics fallback
                physics_probs = np.array([
                    max(row.get("hist_asr_mean", 0), 0) for row in feature_rows
                ])
                if physics_probs.sum() > 0:
                    physics_probs = physics_probs / physics_probs.sum()
                else:
                    physics_probs = np.ones(len(horse_codes)) / len(horse_codes)

                combined = ensemble_predict(
                    lgbm_probs=lgbm_probs, xgb_probs=xgb_probs,
                    ltr_probs=ltr_probs, physics_probs=physics_probs,
                    weights=ensemble_weights,
                )

                predicted_top3 = pick_top3(combined, horse_codes)
                correct = len(set(predicted_top3) & actual_top3_set)

                results.append({
                    "race_id": rid,
                    "date": race_date,
                    "venue": venue,
                    "venue_code": race.get("venue_code", "ST"),
                    "race_no": race.get("race_no", 0),
                    "distance": distance,
                    "condition": race.get("condition", "GOOD"),
                    "field_size": field_size,
                    "actual_top3": json.dumps(actual_top3),
                    "predicted_top3": json.dumps(predicted_top3),
                    "correct_count": correct,
                    "precision_at_3": correct / 3,
                    "hit_any": 1 if correct > 0 else 0,
                    "has_ml_models": bool(ml_models),
                })

                for i, code in enumerate(horse_codes):
                    actual_pos = next(
                        (e["finish_position"] for e in entries if e.get("horse_code") == code), 0
                    )
                    horse_predictions.append({
                        "race_id": rid, "date": race_date,
                        "venue_code": race.get("venue_code", "ST"),
                        "race_no": race.get("race_no", 0),
                        "horse_code": code,
                        "actual_position": actual_pos,
                        "ensemble_prob": round(float(combined[i]), 4),
                        "lgbm_prob": round(float(lgbm_probs[i]), 4) if lgbm_probs is not None else None,
                        "xgb_prob": round(float(xgb_probs[i]), 4) if xgb_probs is not None else None,
                        "ltr_prob": round(float(ltr_probs[i]), 4) if ltr_probs is not None else None,
                        "physics_prob": round(float(physics_probs[i]), 4),
                    })

        # Record all entries into tracker AFTER prediction (no leakage)
        for entry in entries:
            code = entry.get("horse_code", "").strip()
            if not code:
                continue
            re_obj = _entry_to_race_entry(rid, entry, race)
            if re_obj is None:
                continue
            fp = entry.get("finish_position", 0) or 0
            jockey = entry.get("jockey", "")
            tracker.record(code, jockey, re_obj, entry, race, fp, field_size)

    results_df = pd.DataFrame(results)
    predictions_df = pd.DataFrame(horse_predictions)

    if not results_df.empty:
        results_df.to_csv(output_dir / "backtest_results.csv", index=False, encoding="utf-8-sig")
        predictions_df.to_csv(output_dir / "prediction_report.csv", index=False, encoding="utf-8-sig")
        print(f"\nBacktest complete: {len(results_df)} races")
        print(f"  Results: {output_dir / 'backtest_results.csv'}")
        print(f"  Detail:  {output_dir / 'prediction_report.csv'}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run horse racing backtest")
    parser.add_argument("--start", default="2026/01/01")
    parser.add_argument("--end", default="2026/03/26")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--bulk-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_backtest(
        start_date=args.start, end_date=args.end,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        bulk_dir=Path(args.bulk_dir) if args.bulk_dir else None,
        output_dir=Path(args.output) if args.output else None,
    )

    if not results.empty:
        avg_precision = results["precision_at_3"].mean()
        hit_rate = results["hit_any"].mean()
        hit2 = (results["correct_count"] >= 2).mean()
        miss_rate = (results["correct_count"] == 0).mean()
        print(f"\n── Summary ──────────────────────────────────────────")
        print(f"  Races evaluated: {len(results)}")
        print(f"  Avg Precision@3: {avg_precision:.3f}")
        print(f"  Hit Rate (≥1):   {hit_rate:.3f}")
        print(f"  Hit ≥2 Rate:     {hit2:.3f}")
        print(f"  Miss Rate (0/3): {miss_rate:.3f}")


if __name__ == "__main__":
    main()
