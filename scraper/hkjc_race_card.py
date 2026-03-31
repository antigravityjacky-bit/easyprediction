"""
HKJC Race Card Scraper — fetch upcoming race entries from the English race card page.

Scrapes: https://racing.hkjc.com/racing/information/English/Racing/RaceCard.aspx
         ?RaceDate=YYYY/MM/DD&Racecourse=ST&RaceNo=N

Confirmed column layout (from live page 2026/03/29):
    [0]  Horse No.
    [1]  Last 6 Runs
    [2]  Colour (image, skip)
    [3]  Horse (name in text, code in link href horseid=HK_YYYY_CODE)
    [4]  Wt. (draw weight in lbs)
    [5]  Jockey
    [6]  Draw (gate)
    [7]  Trainer
    [8]  Rtg. (rating)
    [9]  Rtg. +/-
    [10] Horse Wt. (Declaration) — in lbs
    [11] Priority
    [12] Gear

Race header from div text:
    "Race 1 - CHUNG CHI ALUMNI HANDICAP"
    "Sunday, March 29, 2026, Sha Tin, 12:45"
    "Turf, \"A+3\" Course, 1400M, Good"

Usage:
    from horseracing.scraper.hkjc_race_card import scrape_race_card, scrape_all_races
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from horseracing.scraper.hkjc_parser import CACHE_DIR, HEADERS, REQUEST_DELAY_SEC

RACE_CARD_URL = (
    "https://racing.hkjc.com/racing/information/English/Racing/RaceCard.aspx"
    "?RaceDate={date}&Racecourse={venue}&RaceNo={race_no}"
)


def _cache_path_en(url: str) -> Path:
    """Cache path for English race card pages."""
    safe = re.sub(r"[^\w\-]", "_", url.replace("https://racing.hkjc.com", ""))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"EN_{safe[:200]}.html"


def _fetch_page(url: str, use_cache: bool = True) -> str:
    """Fetch URL with cache support."""
    cache_file = _cache_path_en(url)
    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    time.sleep(REQUEST_DELAY_SEC)
    headers = dict(HEADERS)
    headers["Accept-Language"] = "en-US,en;q=0.9"
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    html = response.text
    cache_file.write_text(html, encoding="utf-8")
    return html


def _parse_int(text: str, default: int = 0) -> int:
    cleaned = re.sub(r"[^\d]", "", str(text).strip())
    return int(cleaned) if cleaned else default


def _parse_float(text: str, default: float = 0.0) -> float:
    cleaned = re.sub(r"[^\d.]", "", str(text).strip())
    return float(cleaned) if cleaned else default


def _parse_race_header(soup: BeautifulSoup) -> dict:
    """
    Parse race header info: distance, rail, condition.
    Header text example: 'Turf, "A+3" Course, 1400M, Good'
    """
    info = {"distance": 0, "rail": "A", "condition": "GOOD"}
    page_text = soup.get_text(" ", strip=True)

    # Distance: look for NNN(N)M pattern
    m_dist = re.search(r"(\d{3,4})\s*M", page_text, re.IGNORECASE)
    if m_dist:
        info["distance"] = int(m_dist.group(1))

    # Rail: look for "A+3", "B+2", "C", etc. in quotes
    m_rail = re.search(r'"([A-C](?:\+\d)?)"', page_text)
    if m_rail:
        info["rail"] = m_rail.group(1)

    # Condition/Going
    condition_map = {
        "Good to Firm": "GOOD_TO_FIRM",
        "Good to Yielding": "GOOD_TO_YIELDING",
        "Good": "GOOD",
        "Firm": "FIRM",
        "Yielding": "YIELDING",
        "Slow": "SLOW",
        "Heavy": "HEAVY",
    }
    # Match the longest label first to avoid "Good" matching before "Good to Firm"
    for label in sorted(condition_map.keys(), key=len, reverse=True):
        if label.lower() in page_text.lower():
            info["condition"] = condition_map[label]
            break

    return info


def scrape_race_card(date: str, venue: str, race_no: int) -> dict | None:
    """
    Scrape a single race card from HKJC English page.

    Parameters
    ----------
    date : str  — 'YYYY/MM/DD'
    venue : str — 'ST' or 'HV'
    race_no : int

    Returns dict with entries or None if unavailable.
    """
    url = RACE_CARD_URL.format(date=date, venue=venue, race_no=race_no)
    try:
        html = _fetch_page(url, use_cache=True)
    except Exception as e:
        print(f"  ⚠ Failed to fetch race card R{race_no}: {e}")
        return None

    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text(" ", strip=True)
    if "No race meeting" in page_text or len(page_text) < 200:
        return None

    header = _parse_race_header(soup)
    entries = []

    # Find the main race card table
    tables = soup.find_all("table")
    entry_table = None
    header_row = None
    
    # Let's find the header row first
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        
        # Check first 5 rows for the true header
        for r_idx, row in enumerate(rows[:5]):
            cells = row.find_all(["th", "td"])
            if 10 <= len(cells) <= 35:
                first_text = cells[0].get_text(strip=True).lower()
                if first_text == "horse no.":
                    entry_table = table
                    header_row = row
                    # Data rows follow header
                    data_rows = rows[r_idx+1:]
                    break
        if entry_table:
            break

    if not entry_table or not header_row:
        print(f"  ⚠ No entry table found for R{race_no}")
        return None

    # Dynamically find column indices
    col_map = {}
    header_cells = header_row.find_all(["th", "td"])
    for i, cell in enumerate(header_cells):
        text = cell.get_text(" ", strip=True).lower()
        if "horse no" in text: col_map["horse_no"] = i
        elif "last 6 runs" in text: col_map["last_6_runs"] = i
        elif text == "horse": col_map["horse"] = i
        elif "brand no" in text: col_map["brand_no"] = i
        elif text == "wt.": col_map["wt"] = i
        elif "jockey" in text: col_map["jockey"] = i
        elif "draw" in text: col_map["draw"] = i
        elif "trainer" in text: col_map["trainer"] = i
        elif text == "rtg.": col_map["rtg"] = i
        elif "horse wt" in text and "declaration" in text: col_map["horse_wt"] = i
        elif "best time" in text: col_map["best_time"] = i
        elif "age" == text: col_map["age"] = i
        elif "days since last" in text: col_map["days_since_last"] = i

    for row_idx, row in enumerate(data_rows):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
            
        # Fast fail if no horse number
        idx_no = col_map.get("horse_no", 0)
        if idx_no >= len(cells):
            continue
        first_text = cells[idx_no].get_text(strip=True)
        if not first_text or not re.match(r"^\d+$", first_text):
            continue

        try:
            horse_no = int(first_text)

            def get_cell_text(key: str, default: str = "") -> str:
                idx = col_map.get(key)
                if idx is not None and idx < len(cells):
                    return cells[idx].get_text(strip=True)
                return default

            last_6_runs = get_cell_text("last_6_runs")
            
            # Horse name + code
            horse_name = ""
            horse_code = ""
            idx_horse = col_map.get("horse")
            if idx_horse is not None and idx_horse < len(cells):
                horse_cell = cells[idx_horse]
                horse_name = horse_cell.get_text(strip=True)
                link = horse_cell.find("a", href=True)
                if link:
                    m = re.search(r"horseid=HK_\d+_(\w+)", link["href"], re.IGNORECASE)
                    if m:
                        horse_code = m.group(1).upper()
                    link_text = link.get_text(strip=True)
                    if link_text:
                        horse_name = link_text
                        
            # Use brand no column if code not found in link
            if not horse_code:
                horse_code = get_cell_text("brand_no")

            if not horse_code:
                continue

            draw_weight_lb = _parse_float(get_cell_text("wt"))
            
            jockey = get_cell_text("jockey")
            jockey = re.sub(r"\s*\([+-]?\d+\)\s*$", "", jockey).strip()
            
            gate = _parse_int(get_cell_text("draw"))
            trainer = get_cell_text("trainer")
            rating = _parse_int(get_cell_text("rtg"))
            
            horse_weight_lbs = _parse_int(get_cell_text("horse_wt"))
            horse_weight_kg = round(horse_weight_lbs / 2.205) if horse_weight_lbs > 0 else 0
            
            best_time = get_cell_text("best_time")
            age = _parse_int(get_cell_text("age"))
            days_since_last = _parse_int(get_cell_text("days_since_last"))

            entries.append({
                "horse_no": horse_no,
                "horse_code": horse_code,
                "horse_name": horse_name,
                "jockey": jockey,
                "gate": gate,
                "draw_weight_lb": draw_weight_lb,
                "horse_weight_kg": horse_weight_kg,
                "horse_weight_lbs": horse_weight_lbs,
                "rating": rating,
                "last_6_runs": last_6_runs,
                "trainer": trainer,
                "best_time": best_time,
                "age": age,
                "days_since_last": days_since_last,
            })
        except (ValueError, IndexError):
            continue

    if not entries:
        print(f"  ⚠ No entries parsed for R{race_no}")
        return None

    venue_full = "SHA_TIN" if venue == "ST" else "HAPPY_VALLEY"
    return {
        "date": date,
        "venue": venue_full,
        "venue_code": venue,
        "race_no": race_no,
        "distance": header["distance"],
        "condition": header["condition"],
        "rail": header["rail"],
        "entries": entries,
    }


def count_races(date: str, venue: str) -> int:
    """Count how many races exist for a date/venue."""
    count = 0
    for race_no in range(1, 13):
        url = RACE_CARD_URL.format(date=date, venue=venue, race_no=race_no)
        try:
            html = _fetch_page(url, use_cache=True)
            if "No race meeting" in html or len(html) < 500:
                break
            soup = BeautifulSoup(html, "lxml")
            page_text = soup.get_text(" ", strip=True)
            if "Horse No" in page_text and "Jockey" in page_text:
                count += 1
            else:
                break
        except Exception:
            break
    return count


def scrape_all_races(date: str, venue: str) -> list[dict]:
    """Scrape all race cards for a date/venue."""
    races = []
    for race_no in range(1, 13):
        print(f"  Scraping R{race_no}...")
        card = scrape_race_card(date, venue, race_no)
        if card is None:
            break
        races.append(card)
        print(f"    ✓ {len(card['entries'])} horses, {card['distance']}M, {card['condition']}")
    return races


if __name__ == "__main__":
    import json
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2026/03/29"
    venue = sys.argv[2] if len(sys.argv) > 2 else "ST"
    race_no = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"Scraping race card: {date} {venue} R{race_no}")
    card = scrape_race_card(date, venue, race_no)
    if card:
        print(json.dumps(card, indent=2, ensure_ascii=False))
        print(f"\n{len(card['entries'])} horses found")
    else:
        print("No data found")
