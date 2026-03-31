"""
Bulk Race Scraper — collect all HKJC race results for a date range.

Storage layout:
    datasets/raw/bulk_races/
        YYYY-MM-DD_ST_01.json   ← one file per race
        YYYY-MM-DD_HV_03.json
        _index.json             ← master index [{date, venue, race_no, distance, ...}]

Usage (CLI):
    python -m horseracing.scraper.bulk_scraper --start 2026/01/01 --end 2026/03/26

Usage (Python):
    from horseracing.scraper.bulk_scraper import bulk_scrape
    bulk_scrape("2026/01/01", "2026/03/26")
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from horseracing.scraper.hkjc_parser import fetch_page, parse_race_result
from horseracing.scraper.hkjc_sectional import fetch_sectional_times

BULK_DIR = Path(__file__).resolve().parents[3] / "datasets" / "raw" / "bulk_races"
INDEX_PATH = BULK_DIR / "_index.json"
MAX_RACES_PER_DAY = 12   # HKJC rarely runs more than 11 races
REQUEST_DELAY = 2.1      # seconds between non-cached fetches


# ── Helpers ───────────────────────────────────────────────────────────────────

def _race_path(date_str: str, venue: str, race_no: int) -> Path:
    """Return path for one race JSON, e.g. 2026-03-22_ST_03.json"""
    safe_date = date_str.replace("/", "-")
    return BULK_DIR / f"{safe_date}_{venue}_{race_no:02d}.json"


def _load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(index: list[dict]) -> None:
    BULK_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _index_key(date_str: str, venue: str, race_no: int) -> str:
    return f"{date_str}|{venue}|{race_no}"


# ── Core scraping ─────────────────────────────────────────────────────────────

def probe_race_exists(date_str: str, venue: str, race_no: int, refresh: bool = False) -> bool:
    """
    Return True if a race result page for this date/venue/race_no has entries.
    Uses HTML cache unless refresh=True.
    """
    url = (
        f"https://racing.hkjc.com/zh-hk/local/information/localresults"
        f"?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
    )
    try:
        html = fetch_page(url, use_cache=not refresh)
        result = parse_race_result(html)
        return bool(result.get("entries"))
    except Exception:
        return False


def scrape_single_race(date_str: str, venue: str, race_no: int, refresh: bool = False) -> dict | None:
    """
    Fully scrape one race:
      1. parse_race_result()
      2. fetch_sectional_times()

    Returns a race dict or None if the page has no entries.
    """
    url = (
        f"https://racing.hkjc.com/zh-hk/local/information/localresults"
        f"?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
    )
    try:
        html = fetch_page(url, use_cache=not refresh)
        race_info = parse_race_result(html)
    except Exception as exc:
        print(f"    ⚠  parse_race_result failed {date_str} {venue} R{race_no}: {exc}")
        return None

    if not race_info.get("entries"):
        return None

    # Try to fetch sectional times and merge into each entry
    try:
        sectional_map = fetch_sectional_times(date_str, venue, race_no, use_cache=True)
        for entry in race_info["entries"]:
            code = entry.get("horse_code", "")
            if code and code in sectional_map:
                entry["section_times"] = sectional_map[code]
            else:
                entry.setdefault("section_times", [])
    except Exception:
        for entry in race_info["entries"]:
            entry.setdefault("section_times", [])

    return {
        "date":      date_str,
        "venue":     "SHA_TIN" if venue == "ST" else "HAPPY_VALLEY",
        "venue_code": venue,
        "race_no":   race_no,
        "distance":  race_info.get("distance"),
        "condition": race_info.get("condition"),
        "rail":      race_info.get("rail"),
        "entries":   race_info["entries"],
    }


def scrape_race_day(date_str: str, status_cb: Callable[[str], None] | None = None) -> list[dict]:
    """
    Probe and scrape all races for a given date across both ST and HV.
    Returns list of race dicts (may be empty if no races that day).
    """
    races: list[dict] = []
    for venue in ("ST", "HV"):
        for race_no in range(1, MAX_RACES_PER_DAY + 1):
            if not probe_race_exists(date_str, venue, race_no):
                break   # races are numbered consecutively; stop at first gap
            msg = f"  📥 {date_str} {venue} R{race_no}"
            if status_cb:
                status_cb(msg)
            else:
                print(msg)
            race = scrape_single_race(date_str, venue, race_no)
            if race:
                races.append(race)
    return races


# ── Bulk orchestration ────────────────────────────────────────────────────────

def bulk_scrape(
    start_date: str,
    end_date: str,
    skip_cached: bool = True,
    status_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Scrape all races between start_date and end_date (inclusive).
    Format: "YYYY/MM/DD".

    Returns summary dict: {total_days, race_days, total_races, skipped}.
    Each race is saved to its own JSON file; _index.json is updated incrementally.
    """
    BULK_DIR.mkdir(parents=True, exist_ok=True)

    # Build existing index key set for skip logic
    index = _load_index()
    indexed_keys = {_index_key(e["date"], e["venue_code"], e["race_no"]) for e in index}

    # Iterate dates
    start = date.fromisoformat(start_date.replace("/", "-"))
    end   = date.fromisoformat(end_date.replace("/", "-"))
    current = start

    stats = {"total_days": 0, "race_days": 0, "total_races": 0, "skipped": 0}

    while current <= end:
        date_str = current.strftime("%Y/%m/%d")
        stats["total_days"] += 1

        day_races: list[dict] = []
        for venue in ("ST", "HV"):
            for race_no in range(1, MAX_RACES_PER_DAY + 1):
                key = _index_key(date_str, venue, race_no)
                path = _race_path(date_str, venue, race_no)

                today_dt = date.today()
                days_diff = (today_dt - current).days
                # Smart refresh: auto-refresh if date is within last 3 days
                # because results might be finalized after an initial empty check.
                force_refresh = skip_cached is False or (0 <= days_diff <= 3)

                if not probe_race_exists(date_str, venue, race_no, refresh=force_refresh):
                    break  # no more races at this venue today

                msg = f"📥 {date_str} {venue} R{race_no}"
                if status_cb:
                    status_cb(msg)
                else:
                    print(f"  {msg}")

                race = scrape_single_race(date_str, venue, race_no, refresh=force_refresh)
                if not race:
                    break

                # Save race file
                BULK_DIR.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(race, f, ensure_ascii=False, indent=2)

                # Update index
                idx_entry = {
                    "date": race["date"], "venue": race["venue"],
                    "venue_code": race["venue_code"], "race_no": race["race_no"],
                    "distance": race.get("distance"), "condition": race.get("condition"),
                    "rail": race.get("rail"), "n_horses": len(race.get("entries", [])),
                }
                if key not in indexed_keys:
                    index.append(idx_entry)
                    indexed_keys.add(key)
                    _save_index(index)

                day_races.append(race)
                stats["total_races"] += 1

        if day_races:
            stats["race_days"] += 1

        current += timedelta(days=1)

    _save_index(index)
    return stats


def load_bulk_index() -> list[dict]:
    """Return the master index of all scraped races."""
    return _load_index()


# ── Horse History Scraping ───────────────────────────────────────────────────

HISTORY_DIR = Path(__file__).resolve().parents[3] / "datasets" / "raw" / "horse_histories"


def _collect_horse_codes_from_bulk() -> set[str]:
    """Scan all bulk race JSONs and return unique horse codes."""
    codes: set[str] = set()
    if not BULK_DIR.exists():
        return codes
    for path in BULK_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                race = json.load(f)
            for entry in race.get("entries", []):
                code = entry.get("horse_code", "").strip()
                if code:
                    codes.add(code)
        except Exception:
            continue
    return codes


def scrape_horse_histories(
    n_races: int = 6,
    skip_cached: bool = True,
    status_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    For every horse appearing in bulk races, fetch their last n_races
    from HKJC and save to datasets/raw/horse_histories/{code}.json.

    Returns summary dict: {total_horses, scraped, skipped, failed}.
    """
    from horseracing.scraper.hkjc_parser import fetch_last_n_races

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    codes = _collect_horse_codes_from_bulk()

    stats = {"total_horses": len(codes), "scraped": 0, "skipped": 0, "failed": 0}

    for code in sorted(codes):
        path = HISTORY_DIR / f"{code}.json"
        if skip_cached and path.exists():
            stats["skipped"] += 1
            continue

        msg = f"📥 History: {code} (last {n_races} races)"
        if status_cb:
            status_cb(msg)
        else:
            print(f"  {msg}")

        try:
            races = fetch_last_n_races(code, n=n_races)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"horse_code": code, "races": races}, f,
                          ensure_ascii=False, indent=2)
            stats["scraped"] += 1
        except Exception as exc:
            print(f"    ⚠  Failed {code}: {exc}")
            stats["failed"] += 1

    return stats


def load_horse_history(horse_code: str) -> list[dict]:
    """Load cached horse history. Returns list of race dicts or empty list."""
    path = HISTORY_DIR / f"{horse_code.upper()}.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("races", [])
    except Exception:
        return []


def load_race(date_str: str, venue: str, race_no: int) -> dict | None:
    """Load a previously scraped race dict. venue = 'ST' or 'HV'."""
    path = _race_path(date_str, venue, race_no)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk HKJC race scraper")
    parser.add_argument("--start", required=True, help="Start date YYYY/MM/DD")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY/MM/DD)")
    parser.add_argument("--refresh", action="store_true", help="Ignore HTML cache and re-probe recent dates")
    parser.add_argument("--no-skip-cache", action="store_true",
                        help="Re-scrape even if JSON file already exists")
    parser.add_argument("--histories", action="store_true",
                        help="Also scrape horse histories (last 6 races per horse)")
    parser.add_argument("--history-races", type=int, default=6,
                        help="Number of past races to fetch per horse (default: 6)")
    args = parser.parse_args()

    print(f"\n── Bulk Scraper ─────────────────────────────────────────")
    print(f"  Range : {args.start} → {args.end}")
    print(f"  Output: {BULK_DIR}\n")

    stats = bulk_scrape(
        args.start, args.end,
        skip_cached=not args.no_skip_cache,
    )

    print(f"\n── Done ─────────────────────────────────────────────────")
    print(f"  Days scanned : {stats['total_days']}")
    print(f"  Race days    : {stats['race_days']}")
    print(f"  Races scraped: {stats['total_races']}")
    print(f"  Skipped(cache): {stats['skipped']}\n")

    if args.histories:
        print(f"\n── Horse History Scraper ─────────────────────────────────")
        h_stats = scrape_horse_histories(
            n_races=args.history_races,
            skip_cached=not args.no_skip_cache,
        )
        print(f"\n── History Done ─────────────────────────────────────────")
        print(f"  Total horses : {h_stats['total_horses']}")
        print(f"  Scraped      : {h_stats['scraped']}")
        print(f"  Skipped(cache): {h_stats['skipped']}")
        print(f"  Failed       : {h_stats['failed']}\n")


if __name__ == "__main__":
    main()
