"""
HKJC Sectional Times Parser
Handles the separate HTTP request needed to fetch sectional (split) times.

On the HKJC website, sectional times are NOT shown on the main results page —
they load via a secondary request when the user clicks "Sectional Times".

Known URL patterns (HKJC):
  Race result (EN): https://racing.hkjc.com/en-us/local/information/localresults
                      ?RaceDate=YYYY/MM/DD&Racecourse=ST&RaceNo=N
  Sectional  (EN): https://racing.hkjc.com/en-us/local/information/displaysectionaltime
                      ?racedate=DD/MM/YYYY&RaceNo=N
  Sectional (ASPX): https://racing.hkjc.com/racing/information/English/Racing/DisplaySectionalTime.aspx
                      ?RaceDate=DD/MM/YYYY&RaceNo=N
  Past perf      : https://www.hkjc.com/english/racing/horserace.aspx?HorseId=HK_YYYY_XXXX

The dedicated sectional page uses DD/MM/YYYY date format (not YYYY/MM/DD).
"""

from __future__ import annotations


import logging
import re
import time
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString

from horseracing.scraper.hkjc_parser import HEADERS, CACHE_DIR, REQUEST_DELAY_SEC, fetch_page

log = logging.getLogger(__name__)

HKJC_BASE = "https://racing.hkjc.com"

# Primary: MVC-style URL (English)
SECTIONAL_URL_TEMPLATE = (
    HKJC_BASE
    + "/en-us/local/information/displaysectionaltime"
    + "?racedate={date_dmy}&RaceNo={race_no}"
)

# Fallback: ASP.NET WebForms URL (also English)
SECTIONAL_ASPX_URL_TEMPLATE = (
    HKJC_BASE
    + "/racing/information/English/Racing/DisplaySectionalTime.aspx"
    + "?RaceDate={date_dmy}&RaceNo={race_no}"
)

RESULT_URL_TEMPLATE = (
    HKJC_BASE
    + "/en-us/local/information/localresults"
    + "?RaceDate={date}&Racecourse={venue_code}&RaceNo={race_no}"
)

# Chinese results page (sometimes more reliable)
RESULT_ZH_URL_TEMPLATE = (
    HKJC_BASE
    + "/zh-hk/local/information/localresults"
    + "?racedate={date}&Racecourse={venue_code}&RaceNo={race_no}"
)

PAST_PERF_URL_TEMPLATE = (
    "https://www.hkjc.com/english/racing/horserace.aspx?HorseId={horse_id}"
)

VENUE_CODE_MAP = {
    "SHA_TIN":      "ST",
    "HAPPY_VALLEY": "HV",
}

# Regex to pull horse short code from "HORSE NAME (CODE)" or "HORSE NAME\xa0(CODE)"
_CODE_RE = re.compile(r"\(([A-Z]\d{2,4})\)")

# Regex to match a sectional time like "24.57" (but not "1:35.36" total time)
_TIME_RE = re.compile(r"^\d{1,2}\.\d{2}$")


def _to_dmy(date_ymd: str) -> str:
    """Convert 'YYYY/MM/DD' -> 'DD/MM/YYYY' for the sectional time URL."""
    parts = date_ymd.strip("/").split("/")
    if len(parts) == 3 and len(parts[0]) == 4:
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return date_ymd  # already DD/MM/YYYY or unknown


def build_sectional_url(date: str, venue: str, race_no: int) -> str:
    """
    Build the URL for the HKJC sectional times page.

    Parameters
    ----------
    date : str
        Race date in "YYYY/MM/DD" format.
    venue : str
        "SHA_TIN", "HAPPY_VALLEY", "ST", or "HV".
    race_no : int
        Race number on that day (1-based).
    """
    return SECTIONAL_URL_TEMPLATE.format(
        date_dmy=_to_dmy(date), race_no=race_no
    )


def build_sectional_aspx_url(date: str, venue: str, race_no: int) -> str:
    """Build the fallback ASPX-style URL for HKJC sectional times."""
    return SECTIONAL_ASPX_URL_TEMPLATE.format(
        date_dmy=_to_dmy(date), race_no=race_no
    )


def build_result_url(date: str, venue: str, race_no: int) -> str:
    code = VENUE_CODE_MAP.get(venue, venue)
    return RESULT_URL_TEMPLATE.format(
        date=date, venue_code=code, race_no=race_no
    )


def fetch_sectional_times(
    date: str,
    venue: str,
    race_no: int,
    use_cache: bool = True,
) -> dict[str, list[float]]:
    """
    Fetch sectional times for all horses in one race.

    Parameters
    ----------
    date : str
        Race date "YYYY/MM/DD".
    venue : str
        "SHA_TIN", "HAPPY_VALLEY", "ST", or "HV".
    race_no : int
        Race number (1-based).
    use_cache : bool
        Use local HTML cache if available.

    Returns
    -------
    dict : { horse_code -> [section_time_1, section_time_2, ...] }
    Empty dict if the page cannot be parsed.
    """
    # Strategy 1: dedicated sectional page (MVC URL)
    url = build_sectional_url(date, venue, race_no)
    try:
        html = fetch_page(url, use_cache=use_cache)
        result = _parse_sectional_page(html)
        if result:
            return result
        log.debug("Primary sectional URL returned parseable page but no data: %s", url)
    except Exception as exc:
        log.debug("Primary sectional URL failed: %s (%s)", url, exc)

    # Strategy 2: dedicated sectional page (ASPX URL)
    aspx_url = build_sectional_aspx_url(date, venue, race_no)
    try:
        html = fetch_page(aspx_url, use_cache=use_cache)
        result = _parse_sectional_page(html)
        if result:
            return result
        log.debug("ASPX sectional URL returned parseable page but no data: %s", aspx_url)
    except Exception as exc:
        log.debug("ASPX sectional URL failed: %s (%s)", aspx_url, exc)

    # Strategy 3: fetch the results page, extract the sectional link, follow it
    for tmpl in (RESULT_URL_TEMPLATE, RESULT_ZH_URL_TEMPLATE):
        code = VENUE_CODE_MAP.get(venue, venue)
        result_url = tmpl.format(date=date, venue_code=code, race_no=race_no)
        try:
            result = _parse_sectional_from_results(result_url, use_cache=use_cache)
            if result:
                return result
        except Exception as exc:
            log.debug("Results-page fallback failed for %s: %s", result_url, exc)

    log.warning(
        "All sectional time strategies failed for %s %s R%d", date, venue, race_no
    )
    return {}


def fetch_sectional_from_past_perf(
    horse_id: str,
    use_cache: bool = True,
) -> list[dict]:
    """
    Fetch sectional times from the horse's past-performance page.
    Returns list of dicts: [{date, race_no, section_times, position_calls}, ...]
    """
    url = PAST_PERF_URL_TEMPLATE.format(horse_id=horse_id)
    try:
        html = fetch_page(url, use_cache=use_cache)
        return _parse_past_perf_sectionals(html)
    except Exception as exc:
        log.debug("Past-perf sectional fetch failed for %s: %s", horse_id, exc)
        return []


# -- Private parsers -----------------------------------------------------------

def _extract_direct_text(tag) -> str:
    """Get only direct text children of a tag, ignoring nested elements."""
    parts = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
    return "".join(parts).strip()


def _parse_sectional_page(html: str) -> dict[str, list[float]]:
    """
    Parse the dedicated HKJC displaysectionaltime page.

    The page wraps content in ``div.dispalySectionalTime`` (note HKJC's typo).
    Inside, one or more ``table.race_table`` elements (one per race when viewing
    "All").  Each table has:
      - thead: "Finishing Order", "Horse No.", "Horse", then section headers
        ("1st Sec.", "2nd Sec.", ...), then "Time"
      - tbody rows per horse:
          td: finishing_order
          td: horse_no
          td: horse name link  (e.g. <a ...>IRON LEGION&nbsp;(J459)</a>)
          td: section 1  -- contains:
                <p class="f_clear"><span class="f_fl">POS</span><i>MARGIN</i></p>
                <p>24.57</p>
              OR for 200m sub-splits:
                <p class="f_clear"><span class="f_fl">POS</span><i>MARGIN</i></p>
                <p class="sectional_200">23.99
                    <span class="color_blue2">
                        <span>12.08</span>&nbsp;&nbsp;<span>11.91</span>
                    </span>
                </p>
          td: section 2 ...
          td: (blank sections have only <img> placeholder)
          td: total time "1:35.36"

    Horse code is extracted from the link text, e.g. "IRON LEGION (J459)" -> "J459".

    Returns {horse_code: [split_time_1, split_time_2, ...]}
    """
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, list[float]] = {}

    # Find race tables -- look for table.race_table first, then fall back to
    # tables inside div.dispalySectionalTime
    race_tables = soup.find_all("table", class_=re.compile(r"race_table"))
    if not race_tables:
        wrapper = soup.find("div", class_=re.compile(r"dispalySectionalTime"))
        if wrapper:
            race_tables = wrapper.find_all("table")
    if not race_tables:
        # Last resort: find tables that have thead with "Sectional Time" text
        for table in soup.find_all("table"):
            thead = table.find("thead")
            if thead and "Sectional Time" in thead.get_text():
                race_tables.append(table)

    for table in race_tables:
        # Determine number of header columns to figure out section count
        tbody = table.find("tbody")
        if not tbody:
            # Some pages don't use <tbody> -- rows are direct children of <table>
            rows = table.find_all("tr")
            # Skip header rows (those in <thead> or containing <th>)
            data_rows = [
                r for r in rows
                if not r.find_parent("thead") and not r.find("th")
            ]
        else:
            data_rows = tbody.find_all("tr")

        for row in data_rows:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            # Cell 2 (0-indexed) contains horse name + code link
            horse_cell = cells[2]
            link = horse_cell.find("a")
            if not link:
                continue
            link_text = link.get_text(strip=True)
            m = _CODE_RE.search(link_text)
            if not m:
                continue
            horse_code = m.group(1)

            # Section time cells: cells[3] through cells[-2] (last cell is total time)
            # But we also handle cells[3:] since blank cells and the total-time
            # cell won't match the time regex.
            section_times: list[float] = []
            for sec_cell in cells[3:]:
                time_val = _extract_section_time(sec_cell)
                if time_val is not None:
                    section_times.append(time_val)

            if horse_code and section_times:
                result[horse_code] = section_times

    return result


def _extract_section_time(td) -> Optional[float]:
    """
    Extract the sectional time float from a single <td> cell.

    Returns None if the cell doesn't contain a valid sectional time
    (e.g. blank cells with just an <img>, or the total-time cell "1:35.36").
    """
    paragraphs = td.find_all("p")
    for p in paragraphs:
        # Skip position/margin paragraphs (contain <span class="f_fl">)
        if p.find("span", class_="f_fl"):
            continue

        # Get direct text of this <p>, ignoring child <span> elements.
        # For <p class="sectional_200">24.42 <span>...</span></p>, we want "24.42".
        # For <p>22.38</p>, we want "22.38".
        raw = _extract_direct_text(p)
        if _TIME_RE.match(raw):
            try:
                return float(raw)
            except ValueError:
                continue

    # Fallback: check if the cell itself (without sub-elements) has a time
    # This handles cases where time is in the td directly, not in a <p>
    raw = _extract_direct_text(td)
    if _TIME_RE.match(raw):
        try:
            return float(raw)
        except ValueError:
            pass

    return None


def _parse_sectional_from_results(
    result_url: str,
    use_cache: bool = True,
) -> dict[str, list[float]]:
    """
    Fallback: fetch the race results page, find the "Sectional Time" button/link,
    follow it to the dedicated sectional page, and parse that.

    The HKJC results page contains a link like:
      <p class="sectional_time_btn ...">
        <a href="/en-us/local/information/displaysectionaltime?racedate=DD/MM/YYYY&RaceNo=N">
          <img src="...sectional_time_btn.gif" />
        </a>
      </p>
    """
    html = fetch_page(result_url, use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")

    # Look for the sectional time button link
    sectional_link = None

    # Method 1: find <p class="sectional_time_btn"> -> <a>
    btn = soup.find("p", class_=re.compile(r"sectional_time_btn"))
    if btn:
        a_tag = btn.find("a", href=True)
        if a_tag:
            sectional_link = a_tag["href"]

    # Method 2: find any link containing "displaysectionaltime" in href
    if not sectional_link:
        a_tag = soup.find("a", href=re.compile(r"displaysectionaltime", re.I))
        if a_tag:
            sectional_link = a_tag["href"]

    # Method 3: find any link containing "DisplaySectionalTime" (ASPX style)
    if not sectional_link:
        a_tag = soup.find("a", href=re.compile(r"DisplaySectionalTime", re.I))
        if a_tag:
            sectional_link = a_tag["href"]

    if not sectional_link:
        log.debug("No sectional time link found on results page: %s", result_url)
        return {}

    # Resolve relative URL
    if sectional_link.startswith("/"):
        sectional_link = HKJC_BASE + sectional_link

    log.debug("Following sectional link from results page: %s", sectional_link)

    try:
        sec_html = fetch_page(sectional_link, use_cache=use_cache)
        return _parse_sectional_page(sec_html)
    except Exception as exc:
        log.debug("Failed to fetch/parse sectional link %s: %s", sectional_link, exc)
        return {}


def _parse_past_perf_sectionals(html: str) -> list[dict]:
    """
    Parse sectional data from horse's past-performance page.
    HKJC past-perf pages list recent races with split times per section.
    """
    soup = BeautifulSoup(html, "lxml")
    races = []

    perf_table = soup.find("table", class_=re.compile(r"(?i)performance|pastrace"))
    if not perf_table:
        return races

    for row in perf_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        try:
            section_times = []
            position_calls = []

            for cell in cells:
                txt = cell.get_text(strip=True)
                if _TIME_RE.match(txt):
                    section_times.append(float(txt))
                elif re.match(r"^\d{1,2}$", txt) and 1 <= int(txt) <= 20:
                    position_calls.append(int(txt))

            if section_times:
                races.append({
                    "date": cells[0].get_text(strip=True),
                    "section_times": section_times,
                    "position_calls": position_calls,
                })
        except (ValueError, IndexError):
            continue

    return races
