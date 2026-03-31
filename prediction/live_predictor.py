"""
Live Predictor Pipeline — March 29, 2026 Sha Tin Races

Usage:
    python -m horseracing.prediction.live_predictor 2026/03/29 ST 1

Pipeline steps:
1. Load historical bulk data, process it via RollingTracker.
2. Scrape the upcoming target race card.
3. Build features for each horse based on RollingTracker stats.
4. Load ML models and run ensemble predictions.
5. Output final ranking and probabilities.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Suppress lightgbm warnings
warnings.filterwarnings("ignore", category=UserWarning)

from horseracing.scraper.hkjc_race_card import scrape_race_card
from horseracing.ml.dataset import RollingTracker, _entry_to_race_entry, _safe_float
from horseracing.features.engineer import FEATURE_NAMES, VENUE_ENC, CONDITION_ENC, TREND_ENC, STYLE_ENC
from horseracing.backtest.runner import _build_features_from_tracker
from horseracing.ml.models import load_models
from horseracing.ml.ensemble import ensemble_predict

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BULK_DIR = PROJECT_ROOT.parent / "datasets" / "raw" / "bulk_races"
PROCESSED_DIR = PROJECT_ROOT.parent / "datasets" / "processed"

# Current optimal ensemble weights
WEIGHTS = {
    "lgbm": 0.1,
    "xgb": 0.0,
    "ltr": 0.4,
    "physics": 0.5
}


def load_bulk_history() -> RollingTracker:
    """Load all historical bulk races chronologically to populate the RollingTracker."""
    tracker = RollingTracker()
    
    # We always fallback directly to the ml_dataset.csv if it exists!
    # It is simply much faster and less brittle than reading 600 JSONs.
    dataset_path = PROCESSED_DIR / "ml_dataset.csv"
    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset not found: {dataset_path}")
        print("Please run the dataset compiler to build history.")
        return tracker
        
    print(f"Loading history from pre-built dataset {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df.sort_values(by=["date", "race_no"], inplace=True)
    
    # Process row by row
    races_processed = len(df["race_id"].unique())
    
    for _, row in df.iterrows():
        horse_code = row["horse_code"]
        jockey = row["jockey"]
        
        try:
            sec_times = json.loads(row["section_times_json"]) if pd.notna(row.get("section_times_json")) else []
            pos_calls = json.loads(row["position_calls_json"]) if pd.notna(row.get("position_calls_json")) else []
        except:
            sec_times, pos_calls = [], []
            
        race_dict = {
            "date": row["date"],
            "venue": row["venue"],
            "venue_code": row["venue_code"],
            "race_no": int(row["race_no"]),
            "distance": int(row["distance"]),
            "condition": row["condition"],
            "rail": row["rail"]
        }
        
        entry_dict = {
            "gate": int(row["gate"]),
            "draw_weight_lb": float(row["draw_weight_lb"]),
            "horse_weight_kg": float(row["horse_weight_kg"]) if pd.notna(row.get("horse_weight_kg")) else 0,
            "finish_time": float(row["finish_time"]) if pd.notna(row.get("finish_time")) else 0,
            "finish_position": int(row["finish_position"]) if pd.notna(row.get("finish_position")) else 0,
            "section_times": sec_times,
            "position_calls": pos_calls,
            "jockey": jockey
        }
        
        try:
            race_id = f"{row['date']}_{row['venue_code']}_{int(row['race_no']):02d}"
            re_obj = _entry_to_race_entry(race_id, entry_dict, race_dict)
            finish_pos = entry_dict["finish_position"]
            tracker.record(
                horse_code=horse_code,
                jockey=jockey,
                race_entry=re_obj,
                entry_dict=entry_dict,
                race_dict=race_dict,
                finish_pos=finish_pos,
                field_size=int(row["field_size"])
            )
        except Exception:
            continue
            
    print(f"Processed {len(df)} entries from {races_processed} races into RollingTracker.")
    return tracker, df


def main():
    parser = argparse.ArgumentParser(description="Live Race Predictor")
    parser.add_argument("date", type=str, help="Race date (YYYY/MM/DD)")
    parser.add_argument("venue", type=str, help="Venue code (ST or HV)")
    parser.add_argument("race_no", type=int, help="Race number (1-12)")
    args = parser.parse_args()
    
    # 1. Initialize and populate RollingTracker from bulk history
    tracker, df_hist = load_bulk_history()
    
    # Check if we have history
    if not tracker.horse_history:
        print("\n[FAILED] Cannot proceed: RollingTracker has no history.")
        print("The bulk dataset is required to generate accurate features.")
        return
        
    # 2. Try to load from history first (for past races)
    hist_entries = df_hist[(df_hist['date'] == args.date) & 
                           (df_hist['venue_code'] == args.venue) & 
                           (df_hist['race_no'] == args.race_no)]
    
    if not hist_entries.empty:
        print(f"--- Found {len(hist_entries)} entries in historical dataset ---")
        raw_df = hist_entries.copy().reset_index(drop=True)
        # Mock a 'card' object for display
        card = {
            "date": args.date,
            "venue": args.venue,
            "race_no": args.race_no,
            "distance": hist_entries.iloc[0].get('distance', 0),
            "condition": hist_entries.iloc[0].get('condition', 'GOOD'),
            "entries": []
        }
    else:
        # Proceed with live scrape for non-historical races
        print(f"\nScraping live race card for {args.date} {args.venue} R{args.race_no}...")
        card = scrape_race_card(args.date, args.venue, args.race_no)
        
        if not card or not card.get("entries"):
            print("[FAILED] Could not scrape race card or no entries found.")
            return

        print(f"Found {len(card['entries'])} declared horses.")
        
        # 3. Feature engineering for live entries
        print("\nBuilding live features from history...")
        entries = card["entries"]
        feature_rows = []
        
        for entry in entries:
            horse_code = entry["horse_code"]
            horse_name = entry["horse_name"]
            
            # Build raw feature row
            feat_dict = _build_features_from_tracker(entry, card, tracker)
            
            # Add identifiers back in
            feat_dict["horse_code"] = horse_code
            feat_dict["horse_name"] = horse_name
            feat_dict["horse_no"] = entry["horse_no"]
            feat_dict["jockey"] = entry["jockey"]
            feat_dict["gate"] = entry["gate"]
            feat_dict["draw_weight_lb"] = entry["draw_weight_lb"]
            
            feature_rows.append(feat_dict)
            
        raw_df = pd.DataFrame(feature_rows)
    
    # 4. Apply standard feature engineering (label encoding, etc.)
    df = raw_df.copy()
    if "venue_enc" not in df.columns:
        df["venue_enc"] = df.get("venue", pd.Series([card["venue"]]*len(df))).map(VENUE_ENC).fillna(0)
    if "condition_enc" not in df.columns:
        df["condition_enc"] = df.get("condition", pd.Series([card["condition"]]*len(df))).map(CONDITION_ENC).fillna(2)
        
    if "racing_style_enc" not in df.columns:
        if "racing_style" in df.columns:
            df["racing_style_enc"] = df["racing_style"].map(STYLE_ENC).fillna(-1)
        else:
            df["racing_style_enc"] = -1
            
    for trend_col in ["hist_asr_trend", "hist_speed_trend", "hist_fap_trend"]:
        if trend_col in df.columns and df[trend_col].dtype == object:
            df[trend_col] = df[trend_col].map(TREND_ENC).fillna(0)
            
    # Extract model features
    feature_data = {}
    for fname in FEATURE_NAMES:
        if fname in df.columns:
            feature_data[fname] = pd.to_numeric(df[fname], errors="coerce").fillna(0).values
        else:
            feature_data[fname] = np.zeros(len(df))
            
    X_live = pd.DataFrame(feature_data, index=df.index)
    
    # 4. Load models and predict
    models_dir = PROCESSED_DIR / "models"
    print(f"\nLoading models from {models_dir}...")
    try:
        models = load_models(models_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return
        
    try:
        from horseracing.ml.models import predict_top3_prob, predict_rank_scores
        
        lgbm_probs = predict_top3_prob(models["lgbm_top3"], X_live) if "lgbm_top3" in models else None
        xgb_probs = predict_top3_prob(models["xgb_top3"], X_live) if "xgb_top3" in models else None
        ltr_scores = predict_rank_scores(models["ltr_ranker"], X_live) if "ltr_ranker" in models else None
        
        # Softmax approximation for LTR if present
        if ltr_scores is not None:
            exps = np.exp(ltr_scores - np.max(ltr_scores))
            ltr_probs = exps / np.sum(exps)
        else:
            ltr_probs = None

        print(f"Using ensemble weights: {WEIGHTS}")
        final_probs = ensemble_predict(
            lgbm_probs=lgbm_probs,
            xgb_probs=xgb_probs,
            ltr_probs=ltr_probs,
            physics_probs=None,
            weights=WEIGHTS,
        )
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return
        
    # 5. Output results
    final_df = raw_df[['horse_no', 'horse_name', 'jockey', 'gate', 'draw_weight_lb']].copy()
    final_df['prob'] = final_probs
    if 'asr' in df.columns: final_df['asr'] = df['asr'].fillna(0)
    if 'fap' in df.columns: final_df['fap'] = df['fap'].fillna(0)
    if 'hist_win_rate' in df.columns: final_df['hist_win_rate'] = df['hist_win_rate'].fillna(0)

    # Sort by probability descending
    final_df = final_df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print(f"PREDICTIONS: {args.date} {args.venue} Race {args.race_no}")
    print(f"Distance: {card['distance']}m, Condition: {card['condition']}")
    print(f"{'='*70}")
    
    if len(final_df) < 3:
        print("Not enough horses to form Banker + Legs combo.")
        return
        
    banker = final_df.iloc[0]
    legs = final_df.iloc[1:3]
    others = final_df.iloc[3:]
    
    print(f"👑 [A] BANKER (KEY)")
    print(f"  #{(banker['horse_no']):<2} {banker['horse_name']:<20} ({banker['jockey']:<12}) Prob: {banker['prob']:.1%}")
    print("\n🐎 [B,C] LEGS (SUBS)")
    for i, row in legs.iterrows():
        print(f"  #{row['horse_no']:<2} {row['horse_name']:<20} ({row['jockey']:<12}) Prob: {row['prob']:.1%}")
        
    print("\n📈 QUINELLA COVERAGE (A + Leg)")
    p_a = banker['prob']
    p_b, p_c = legs.iloc[0]['prob'], legs.iloc[1]['prob']
    print(f"  A-B Combo Est: {(p_a * p_b)*100:.1f}%")
    print(f"  A-C Combo Est: {(p_a * p_c)*100:.1f}%")
    
    print(f"{'-'*70}")
    print(f"{'ALL REMAINING HORSES':^70}")
    print(f"{'-'*70}")
    print(f"{'RNK':<4} {'#':<3} {'HORSE':<20} {'JOCKEY':<15} {'GATE':<5} {'PROB':<7} {'WARN'}")
    
    for i, row in others.iterrows():
        rank = i + 1 # Use actual 1-based rank in the sorted list
        print(f"{rank:<4} {row['horse_no']:<3} {row['horse_name']:<20} {row['jockey']:<15} {row['gate']:<5} {row['prob']:<7.1%}")
        
    print(f"{'-'*70}")
    print(f"💡 STRATEGY: High-confidence Banker + Selected Legs (Optimized Coverage).")
    print(f"{'='*70}\n") 


if __name__ == "__main__":
    main()
