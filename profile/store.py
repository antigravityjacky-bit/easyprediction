"""
Profile persistence — save/load HorseProfile objects as JSON.
Profiles are stored in datasets/processed/horse_profiles.json
"""

from __future__ import annotations


import json
import os
from dataclasses import asdict
from pathlib import Path

from horseracing.profile.builder import HorseProfile, RaceMetrics, MetricSummary

DEFAULT_STORE_PATH = Path(__file__).resolve().parents[2] / "datasets" / "processed" / "horse_profiles.json"


def _profile_to_dict(profile: HorseProfile) -> dict:
    return {
        "horse_id": profile.horse_id,
        "horse_name": profile.horse_name,
        "recent_races": [asdict(r) for r in profile.recent_races],
        "asr": asdict(profile.asr) if profile.asr else None,
        "true_speed": asdict(profile.true_speed) if profile.true_speed else None,
        "fap": asdict(profile.fap) if profile.fap else None,
        "edi": profile.edi,
        "fi": asdict(profile.fi) if profile.fi else None,
        "pa": profile.pa,
        "preferred_distance_range": list(profile.preferred_distance_range) if profile.preferred_distance_range else None,
        "racing_style": profile.racing_style,
    }


def _dict_to_profile(d: dict) -> HorseProfile:
    races = [RaceMetrics(**r) for r in d["recent_races"]]

    def _to_summary(s):
        if s is None:
            return None
        return MetricSummary(**s)

    profile = HorseProfile(
        horse_id=d["horse_id"],
        horse_name=d["horse_name"],
        recent_races=races,
        asr=_to_summary(d.get("asr")),
        true_speed=_to_summary(d.get("true_speed")),
        fap=_to_summary(d.get("fap")),
        edi=d.get("edi"),
        fi=_to_summary(d.get("fi")),
        pa=d.get("pa"),
        preferred_distance_range=tuple(d["preferred_distance_range"]) if d.get("preferred_distance_range") else None,
        racing_style=d.get("racing_style"),
    )
    return profile


def load_all_profiles(path: Path = DEFAULT_STORE_PATH) -> dict[str, HorseProfile]:
    """Load all profiles from JSON store. Returns {horse_id: HorseProfile}."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: _dict_to_profile(v) for k, v in raw.items()}


def save_all_profiles(profiles: dict[str, HorseProfile], path: Path = DEFAULT_STORE_PATH) -> None:
    """Save all profiles to JSON store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: _profile_to_dict(v) for k, v in profiles.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_profile(horse_id: str, path: Path = DEFAULT_STORE_PATH) -> HorseProfile | None:
    """Load a single profile by horse_id."""
    profiles = load_all_profiles(path)
    return profiles.get(horse_id)


def upsert_profile(profile: HorseProfile, path: Path = DEFAULT_STORE_PATH) -> None:
    """Insert or update a single profile in the store."""
    profiles = load_all_profiles(path)
    profiles[profile.horse_id] = profile
    save_all_profiles(profiles, path)


SCRAPED_RACE_PATH = DEFAULT_STORE_PATH.parent / "scraped_race.json"


def save_scraped_race(race: dict | None) -> None:
    """Persist scraped_race dict to disk so it survives page refreshes."""
    SCRAPED_RACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCRAPED_RACE_PATH, "w", encoding="utf-8") as f:
        json.dump(race or {}, f, ensure_ascii=False, indent=2)


def load_scraped_race() -> dict | None:
    """Load scraped_race from disk. Returns None if not found or empty."""
    if not SCRAPED_RACE_PATH.exists():
        return None
    with open(SCRAPED_RACE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if data else None


BACKTEST_LOG_PATH = DEFAULT_STORE_PATH.parent / "backtest_log.json"


def load_backtest_log() -> list[dict]:
    """Load all saved backtest entries from disk."""
    if not BACKTEST_LOG_PATH.exists():
        return []
    with open(BACKTEST_LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_backtest_entry(entry: dict) -> None:
    """Append one backtest entry to the log, persisting immediately."""
    log = load_backtest_log()
    log.append(entry)
    BACKTEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BACKTEST_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def clear_backtest_log() -> None:
    """Wipe the backtest log."""
    with open(BACKTEST_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
