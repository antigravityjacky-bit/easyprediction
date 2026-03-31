"""
ML Dataset Builder (v2)
Converts bulk race JSONs into a flat pandas DataFrame with rolling history
features built entirely from the bulk data itself (no external horse_histories).

Key improvements over v1:
  - Builds rolling history from bulk data (each horse's earlier races)
  - Jockey win/place rate features
  - Horse win/place rate and avg finish position
  - Section-time-derived features (with real sectional data)
  - Field-relative features (speed rank, weight rank)

Usage (CLI):
    python -m horseracing.ml.dataset --bulk-dir datasets/raw/bulk_races --output datasets/processed/ml_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from horseracing.scraper.bulk_scraper import BULK_DIR
from horseracing.profile.builder import (
    RaceEntry, compute_metrics, build_profile_from_entries,
)
from horseracing.metrics.advanced import compute_all_advanced

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "datasets" / "processed"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _race_id(date: str, venue_code: str, race_no: int) -> str:
    return f"{date.replace('/', '-')}_{venue_code}_{race_no:02d}"


def _parse_date(date_str: str) -> str:
    return date_str.replace("-", "/")


def _days_between(date1: str, date2: str) -> int:
    try:
        d1 = datetime.strptime(date1.replace("-", "/"), "%Y/%m/%d")
        d2 = datetime.strptime(date2.replace("-", "/"), "%Y/%m/%d")
        return abs((d2 - d1).days)
    except (ValueError, AttributeError):
        return 0


def _safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def _entry_to_race_entry(race_id: str, entry: dict, race: dict) -> RaceEntry | None:
    try:
        finish_time = entry.get("finish_time", 0.0)
        if not finish_time or finish_time <= 0:
            return None
        venue = race.get("venue", "SHA_TIN")
        return RaceEntry(
            race_id=race_id,
            horse_id=entry.get("horse_code", ""),
            date=race.get("date", ""),
            venue=venue,
            rail=race.get("rail", "A"),
            distance=race.get("distance", 0) or 0,
            condition=race.get("condition", "GOOD"),
            gate=entry.get("gate", 0) or 0,
            draw_weight_lb=entry.get("draw_weight_lb", 0.0) or 0.0,
            finish_time=finish_time,
            finish_position=entry.get("finish_position", 0) or 0,
            section_times=entry.get("section_times", []) or [],
            position_calls=entry.get("position_calls", []) or [],
            horse_weight_kg=entry.get("horse_weight_kg", 0) or 0,
            jockey_id=entry.get("jockey", ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


# ── Rolling history tracker ─────────────────────────────────────────────────

class RollingTracker:
    """Tracks per-horse and per-jockey history as races are processed chronologically."""

    def __init__(self):
        # horse_code → list of (race_entry_obj, entry_dict, race_dict)
        self.horse_history: dict[str, list[tuple]] = defaultdict(list)
        # jockey → list of (finish_position, field_size)
        self.jockey_stats: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def get_horse_entries(self, horse_code: str) -> list[RaceEntry]:
        return [t[0] for t in self.horse_history[horse_code]]

    def get_horse_raw(self, horse_code: str) -> list[dict]:
        """Return raw entry+race dicts for history building."""
        results = []
        for re_obj, entry_d, race_d in self.horse_history[horse_code]:
            results.append({
                "date": race_d.get("date", ""),
                "venue": race_d.get("venue", "SHA_TIN"),
                "venue_code": race_d.get("venue_code", "ST"),
                "race_no": race_d.get("race_no", 0),
                "distance": race_d.get("distance", 0),
                "condition": race_d.get("condition", "GOOD"),
                "rail": race_d.get("rail", "A"),
                "gate": entry_d.get("gate", 0),
                "draw_weight_lb": entry_d.get("draw_weight_lb", 0),
                "horse_weight_kg": entry_d.get("horse_weight_kg", 0),
                "finish_time": entry_d.get("finish_time", 0),
                "finish_position": entry_d.get("finish_position", 0),
                "lbw": entry_d.get("lbw", 0.0),
                "section_times": entry_d.get("section_times", []),
                "position_calls": entry_d.get("position_calls", []),
                "jockey": entry_d.get("jockey", ""),
                "stewards_note": entry_d.get("stewards_note", ""),
            })
        return results

    def get_jockey_stats(self, jockey: str) -> dict:
        records = self.jockey_stats.get(jockey, [])
        if not records:
            return {"jockey_win_rate": 0, "jockey_place_rate": 0, "jockey_n_rides": 0}
        n = len(records)
        wins = sum(1 for fp, fs in records if fp == 1)
        places = sum(1 for fp, fs in records if fp <= 3)
        return {
            "jockey_win_rate": wins / n,
            "jockey_place_rate": places / n,
            "jockey_n_rides": n,
        }

    def record(self, horse_code: str, jockey: str, race_entry: RaceEntry,
               entry_dict: dict, race_dict: dict, finish_pos: int, field_size: int):
        self.horse_history[horse_code].append((race_entry, entry_dict, race_dict))
        if jockey:
            self.jockey_stats[jockey].append((finish_pos, field_size))


# ── Core dataset building ────────────────────────────────────────────────────

def build_dataset(
    bulk_dir: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build ML-ready dataset with rolling history from bulk races.
    Processes races chronologically so each horse's features use only prior data.
    """
    bulk_dir = bulk_dir or BULK_DIR
    rows: list[dict] = []

    # Load and sort all races chronologically
    race_files = sorted(bulk_dir.glob("*.json"))
    race_files = [f for f in race_files if not f.name.startswith("_")]

    all_races = []
    for file_path in race_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                race = json.load(f)
            all_races.append(race)
        except Exception:
            continue

    # Sort by date then venue then race_no
    all_races.sort(key=lambda r: (
        _parse_date(r.get("date", "")),
        r.get("venue_code", ""),
        r.get("race_no", 0),
    ))

    print(f"Building dataset from {len(all_races)} races...")

    tracker = RollingTracker()

    for race_i, race in enumerate(all_races):
        race_date = _parse_date(race.get("date", ""))
        venue_code = race.get("venue_code", "ST")
        venue = race.get("venue", "SHA_TIN")
        race_no = race.get("race_no", 0)
        rid = _race_id(race_date, venue_code, race_no)
        distance = race.get("distance", 0) or 0
        condition = race.get("condition", "GOOD")
        rail = race.get("rail", "A")
        entries = race.get("entries", [])
        field_size = len(entries)

        # Collect field-level data for relative features
        field_weights = [_safe_float(e.get("draw_weight_lb")) for e in entries
                         if _safe_float(e.get("draw_weight_lb")) > 0]
        avg_field_weight = sum(field_weights) / len(field_weights) if field_weights else 0

        # Process each horse
        race_rows = []
        for entry in entries:
            horse_code = entry.get("horse_code", "").strip()
            if not horse_code:
                continue

            race_entry = _entry_to_race_entry(rid, entry, race)
            if race_entry is None:
                continue

            # Compute current race metrics
            try:
                metrics = compute_metrics(race_entry)
            except Exception:
                continue

            finish_pos = entry.get("finish_position", 0) or 0
            jockey = entry.get("jockey", "")

            row = {
                # Identifiers
                "race_id": rid,
                "date": race_date,
                "venue": venue,
                "venue_code": venue_code,
                "race_no": race_no,
                "horse_no": entry.get("horse_no", 0) or 0,
                "horse_code": horse_code,
                "horse_name": entry.get("horse_name", ""),

                # Race context
                "distance": distance,
                "condition": condition,
                "rail": rail,
                "field_size": field_size,

                # Horse-in-race data
                "gate": entry.get("gate", 0) or 0,
                "draw_weight_lb": entry.get("draw_weight_lb", 0.0) or 0.0,
                "horse_weight_kg": entry.get("horse_weight_kg", 0) or 0,
                "jockey": jockey,
                "finish_position": finish_pos,
                "lbw": entry.get("lbw", 0.0) or 0.0,
                "finish_time": entry.get("finish_time", 0.0) or 0.0,

                # Current race metrics
                "asr": metrics.asr,
                "true_speed_ms": metrics.true_speed_ms,
                "fap": metrics.fap,
                "edi": metrics.edi,
                "fi": metrics.fi,
                "pa": metrics.pa,

                # Section times metadata
                "has_section_times": 1 if (entry.get("section_times") and
                                           len(entry.get("section_times", [])) >= 2) else 0,
                "n_sections": len(entry.get("section_times", []) or []),

                # Section times stored for CSV
                "section_times_json": json.dumps(entry.get("section_times", [])),
                "position_calls_json": json.dumps(entry.get("position_calls", [])),
                "stewards_note": entry.get("stewards_note", ""),

                # Target
                "top3": 1 if finish_pos <= 3 else 0,

                # Weight relative to field
                "weight_vs_field_avg": (_safe_float(entry.get("draw_weight_lb")) -
                                        avg_field_weight),
            }

            # ── History-based features (from rolling tracker) ────────
            hist_entries = tracker.get_horse_entries(horse_code)
            hist_raw = tracker.get_horse_raw(horse_code)
            n_hist = len(hist_entries)

            if n_hist > 0:
                # Days since last race
                row["days_since_last"] = _days_between(hist_entries[-1].date, race_date)

                # Horse performance stats
                hist_fps = [e.finish_position for e in hist_entries if e.finish_position > 0]
                if hist_fps:
                    row["hist_avg_finish_pos"] = sum(hist_fps) / len(hist_fps)
                    row["hist_best_finish_pos"] = min(hist_fps)
                    row["hist_win_rate"] = sum(1 for fp in hist_fps if fp == 1) / len(hist_fps)
                    row["hist_place_rate"] = sum(1 for fp in hist_fps if fp <= 3) / len(hist_fps)
                else:
                    row["hist_avg_finish_pos"] = 7
                    row["hist_best_finish_pos"] = 14
                    row["hist_win_rate"] = 0
                    row["hist_place_rate"] = 0

                # Build profile from last 6 races (was 3 — more data = better signal)
                try:
                    profile = build_profile_from_entries(
                        horse_code, horse_code, hist_entries[-6:]
                    )
                    if profile.asr:
                        row["hist_asr_mean"] = profile.asr.mean
                        row["hist_asr_std"] = profile.asr.std
                        row["hist_asr_trend"] = {"improving": 1, "stable": 0, "declining": -1}.get(profile.asr.trend, 0)
                    if profile.true_speed:
                        row["hist_speed_mean"] = profile.true_speed.mean
                        row["hist_speed_std"] = profile.true_speed.std
                        row["hist_speed_trend"] = {"improving": 1, "stable": 0, "declining": -1}.get(profile.true_speed.trend, 0)
                    if profile.fap:
                        row["hist_fap_mean"] = profile.fap.mean
                        row["hist_fap_std"] = profile.fap.std
                        row["hist_fap_trend"] = {"improving": 1, "stable": 0, "declining": -1}.get(profile.fap.trend, 0)
                    if profile.fi:
                        row["hist_fi_mean"] = profile.fi.mean
                        row["hist_fi_std"] = profile.fi.std
                    row["hist_edi_mean"] = _safe_float(profile.edi)
                    row["hist_pa_mean"] = _safe_float(profile.pa)
                    row["racing_style"] = profile.racing_style or "unknown"
                except Exception:
                    pass

                # Advanced metrics from most recent history race
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
                        days_since_last=row.get("days_since_last", -1),
                        recent_asr_values=recent_asrs,
                        history=hist_raw,
                    )
                    row["peak_speed_ms"] = _safe_float(adv.get("peak_speed_ms"))
                    row["finishing_burst"] = _safe_float(adv.get("finishing_burst"), 1.0)
                    row["speed_decay_rate"] = _safe_float(adv.get("speed_decay_rate"))
                    row["max_acceleration"] = _safe_float(adv.get("max_acceleration"))
                    row["kinetic_energy_index"] = _safe_float(adv.get("kinetic_energy_index"))
                    row["weight_efficiency"] = _safe_float(adv.get("weight_efficiency"))
                    row["power_output_watts"] = _safe_float(adv.get("power_output_watts"))
                    row["turn_penalty_ms"] = _safe_float(adv.get("turn_penalty_ms"))
                    row["draw_extra_distance_m"] = _safe_float(adv.get("draw_extra_distance_m"))
                    row["positioning_cost"] = _safe_float(adv.get("positioning_cost"))
                    row["drafting_factor"] = _safe_float(adv.get("drafting_factor"))
                    row["form_trend"] = _safe_float(adv.get("form_trend"))
                    row["freshness"] = _safe_float(adv.get("freshness"), 0.5)
                    row["distance_aptitude_ms"] = _safe_float(adv.get("distance_aptitude_ms"))
                    row["track_affinity_ms"] = _safe_float(adv.get("track_affinity_ms"))
                except Exception:
                    pass

                # Venue and distance history counts
                venue_races = [e for e in hist_entries if e.venue == venue]
                row["n_races_same_venue"] = len(venue_races)
                if venue_races:
                    venue_fps = [e.finish_position for e in venue_races if e.finish_position > 0]
                    row["venue_avg_finish"] = sum(venue_fps) / len(venue_fps) if venue_fps else 7
                    row["venue_place_rate"] = sum(1 for fp in venue_fps if fp <= 3) / len(venue_fps) if venue_fps else 0

                dist_races = [e for e in hist_entries if abs(e.distance - distance) <= 200]
                row["n_races_similar_dist"] = len(dist_races)
                if dist_races:
                    dist_fps = [e.finish_position for e in dist_races if e.finish_position > 0]
                    row["dist_avg_finish"] = sum(dist_fps) / len(dist_fps) if dist_fps else 7
                    row["dist_place_rate"] = sum(1 for fp in dist_fps if fp <= 3) / len(dist_fps) if dist_fps else 0

                # Last race performance
                row["last_finish_pos"] = hist_entries[-1].finish_position
                row["last_asr"] = 0
                try:
                    last_m = compute_metrics(hist_entries[-1])
                    row["last_asr"] = last_m.asr
                    row["last_speed_ms"] = last_m.true_speed_ms
                    row["last_fap"] = last_m.fap
                except Exception:
                    pass

                # Distance change from last race
                row["dist_change"] = distance - (hist_entries[-1].distance or 0)

                # Weight change from last race
                last_weight = hist_entries[-1].draw_weight_lb
                row["weight_change"] = _safe_float(entry.get("draw_weight_lb")) - last_weight

            else:
                # No history
                row["days_since_last"] = -1
                row["hist_avg_finish_pos"] = 7
                row["hist_best_finish_pos"] = 14
                row["hist_win_rate"] = 0
                row["hist_place_rate"] = 0
                row["n_races_same_venue"] = 0
                row["n_races_similar_dist"] = 0
                row["last_finish_pos"] = 0
                row["last_asr"] = 0
                row["dist_change"] = 0
                row["weight_change"] = 0

            row["n_history_races"] = n_hist

            # ── Jockey features ─────────────────────────────────────
            jstats = tracker.get_jockey_stats(jockey)
            row.update(jstats)

            # ── Interaction features ────────────────────────────────
            row["gate_x_venue"] = row["gate"] * (1 if venue == "HAPPY_VALLEY" else 0)
            row["weight_x_distance"] = _safe_float(entry.get("draw_weight_lb")) * distance / 1000.0

            race_rows.append((row, horse_code, jockey, race_entry, entry, race, finish_pos, field_size))

        # Record all entries in this race AFTER computing features (prevent leakage)
        for row, hc, jk, re_obj, entry_d, race_d, fp, fs in race_rows:
            rows.append(row)
            tracker.record(hc, jk, re_obj, entry_d, race_d, fp, fs)

        if (race_i + 1) % 50 == 0:
            print(f"  Processed {race_i + 1}/{len(all_races)} races, {len(rows)} entries")

    df = pd.DataFrame(rows)

    # Sort by date then race
    if not df.empty:
        df = df.sort_values(["date", "venue_code", "race_no", "finish_position"])
        df = df.reset_index(drop=True)

    # Save
    if output_path is None:
        output_path = PROCESSED_DIR / "ml_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Report stats
    n_with_hist = (df["n_history_races"] > 0).sum() if "n_history_races" in df.columns else 0
    n_with_sect = (df["has_section_times"] > 0).sum() if "has_section_times" in df.columns else 0
    print(f"\nDataset saved: {output_path}")
    print(f"  Rows: {len(df)}, Races: {df['race_id'].nunique()}")
    print(f"  With history: {n_with_hist} ({n_with_hist/len(df)*100:.1f}%)")
    print(f"  With section times: {n_with_sect} ({n_with_sect/len(df)*100:.1f}%)")

    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML dataset from bulk races")
    parser.add_argument("--bulk-dir", default=None,
                        help="Path to bulk_races directory")
    parser.add_argument("--output", default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    bulk_dir = Path(args.bulk_dir) if args.bulk_dir else None
    output = Path(args.output) if args.output else None
    build_dataset(bulk_dir=bulk_dir, output_path=output)


if __name__ == "__main__":
    main()
