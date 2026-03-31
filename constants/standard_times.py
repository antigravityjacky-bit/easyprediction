"""
Standard Times (Par Times) — HKJC Official Reference
Data updated: 2025/08/26

Standard time is defined as the expected win time for a winner on a 'GOOD' track.
Used to calculate race pace severity (Signal B).
"""

from __future__ import annotations

# Standard deviation defaults by distance (if historical SD is unavailable)
STANDARD_DEVIATION_DEFAULTS = {
    1000: 0.8,
    1200: 1.0,
    1400: 1.2,
    1600: 1.3,
    1650: 1.35,
    1800: 1.4,
    2000: 1.5,
    2200: 1.6,
    2400: 1.8,
}

# Values in seconds
# Key: (Venue, Track, Distance, Class) -> time_sec
# Venue: ST, HV
# Track: TURF, AWT
# Class: GP (Group), 1, 2, 3, 4, 5, GRIFFIN
STANDARD_TIMES = {
    # SHA TIN TURF (沙田草地)
    ("ST", "TURF", 1000, "GP"): 55.90,
    ("ST", "TURF", 1000, "2"): 56.05,
    ("ST", "TURF", 1000, "3"): 56.45,
    ("ST", "TURF", 1000, "4"): 56.65,
    ("ST", "TURF", 1000, "5"): 57.00,
    ("ST", "TURF", 1000, "GRIFFIN"): 56.65,

    ("ST", "TURF", 1200, "GP"): 68.15,
    ("ST", "TURF", 1200, "1"): 68.45,
    ("ST", "TURF", 1200, "2"): 68.65,
    ("ST", "TURF", 1200, "3"): 69.00,
    ("ST", "TURF", 1200, "4"): 69.35,
    ("ST", "TURF", 1200, "5"): 69.55,
    ("ST", "TURF", 1200, "GRIFFIN"): 69.90,

    ("ST", "TURF", 1400, "GP"): 81.10,
    ("ST", "TURF", 1400, "1"): 81.25,
    ("ST", "TURF", 1400, "2"): 81.45,
    ("ST", "TURF", 1400, "3"): 81.65,
    ("ST", "TURF", 1400, "4"): 82.00,
    ("ST", "TURF", 1400, "5"): 82.30,

    ("ST", "TURF", 1600, "GP"): 93.90,
    ("ST", "TURF", 1600, "1"): 94.05,
    ("ST", "TURF", 1600, "2"): 94.25,
    ("ST", "TURF", 1600, "3"): 94.70,
    ("ST", "TURF", 1600, "4"): 94.90,
    ("ST", "TURF", 1600, "5"): 95.45,

    ("ST", "TURF", 1800, "GP"): 107.10,
    ("ST", "TURF", 1800, "2"): 107.30,
    ("ST", "TURF", 1800, "3"): 107.50,
    ("ST", "TURF", 1800, "4"): 107.85,
    ("ST", "TURF", 1800, "5"): 108.45,

    ("ST", "TURF", 2000, "GP"): 120.50,
    ("ST", "TURF", 2000, "1"): 121.20,
    ("ST", "TURF", 2000, "2"): 121.70,
    ("ST", "TURF", 2000, "3"): 121.90,
    ("ST", "TURF", 2000, "4"): 122.35,
    ("ST", "TURF", 2000, "5"): 122.65,

    ("ST", "TURF", 2400, "GP"): 147.00,

    # HAPPY VALLEY TURF (跑馬地草地)
    ("HV", "TURF", 1000, "2"): 56.40,
    ("HV", "TURF", 1000, "3"): 56.65,
    ("HV", "TURF", 1000, "4"): 57.20,
    ("HV", "TURF", 1000, "5"): 57.35,

    ("HV", "TURF", 1200, "1"): 69.10,
    ("HV", "TURF", 1200, "2"): 69.30,
    ("HV", "TURF", 1200, "3"): 69.60,
    ("HV", "TURF", 1200, "4"): 69.90,
    ("HV", "TURF", 1200, "5"): 70.10,

    ("HV", "TURF", 1650, "1"): 99.10,
    ("HV", "TURF", 1650, "2"): 99.30,
    ("HV", "TURF", 1650, "3"): 99.90,
    ("HV", "TURF", 1650, "4"): 100.10,
    ("HV", "TURF", 1650, "5"): 100.30,

    ("HV", "TURF", 1800, "GP"): 108.95,
    ("HV", "TURF", 1800, "2"): 109.15,
    ("HV", "TURF", 1800, "3"): 109.45,
    ("HV", "TURF", 1800, "4"): 109.65,
    ("HV", "TURF", 1800, "5"): 109.95,

    ("HV", "TURF", 2200, "3"): 136.60,
    ("HV", "TURF", 2200, "4"): 137.05,
    ("HV", "TURF", 2200, "5"): 137.35,

    # SHA TIN AWT (沙田全天候)
    ("ST", "AWT", 1200, "2"): 68.35,
    ("ST", "AWT", 1200, "3"): 68.55,
    ("ST", "AWT", 1200, "4"): 68.95,
    ("ST", "AWT", 1200, "5"): 69.35,

    ("ST", "AWT", 1650, "1"): 97.80,
    ("ST", "AWT", 1650, "2"): 98.40,
    ("ST", "AWT", 1650, "3"): 98.60,
    ("ST", "AWT", 1650, "4"): 99.05,
    ("ST", "AWT", 1650, "5"): 99.45,

    ("ST", "AWT", 1800, "3"): 108.05,
    ("ST", "AWT", 1800, "4"): 108.55,
    ("ST", "AWT", 1800, "5"): 109.45,
}

# Standard Sectional Splits mapping (Simplified) — 起點-終點 splits
# Key: (Venue, Track, Distance, Class) -> list[splitTime]
SECTIONAL_PARS = {
    ("ST", "TURF", 1000, "GP"): [13.05, 20.60, 22.25],
    ("ST", "TURF", 1000, "2"): [13.10, 20.60, 22.35],
    ("ST", "TURF", 1000, "3"): [13.05, 20.65, 22.75],
    ("ST", "TURF", 1000, "4"): [13.00, 20.75, 22.90],
    ("ST", "TURF", 1000, "5"): [13.15, 20.95, 22.90],
    ("ST", "TURF", 1000, "GRIFFIN"): [13.25, 20.80, 22.60],

    ("ST", "TURF", 1200, "GP"): [23.55, 22.20, 22.40],
    ("ST", "TURF", 1200, "1"): [23.60, 22.25, 22.60],
    ("ST", "TURF", 1200, "2"): [23.75, 22.25, 22.65],
    ("ST", "TURF", 1200, "3"): [23.70, 22.35, 22.95],
    ("ST", "TURF", 1200, "4"): [23.75, 22.45, 23.15],
    ("ST", "TURF", 1200, "5"): [23.85, 22.40, 23.30],
    ("ST", "TURF", 1200, "GRIFFIN"): [23.95, 22.95, 23.00],

    ("ST", "TURF", 1400, "GP"): [13.50, 22.35, 22.85, 22.40],
    ("ST", "TURF", 1400, "1"): [13.65, 22.00, 22.90, 22.70],
    ("ST", "TURF", 1400, "2"): [13.45, 21.90, 23.10, 23.00],
    ("ST", "TURF", 1400, "3"): [13.45, 21.80, 23.15, 23.25],
    ("ST", "TURF", 1400, "4"): [13.45, 21.75, 23.40, 23.40],
    ("ST", "TURF", 1400, "5"): [13.40, 21.90, 23.35, 23.65],

    ("ST", "TURF", 1600, "GP"): [24.85, 23.05, 23.25, 22.75],
    ("ST", "TURF", 1600, "1"): [24.75, 23.15, 23.15, 23.00],
    ("ST", "TURF", 1600, "2"): [24.55, 23.15, 23.45, 23.10],
    ("ST", "TURF", 1600, "3"): [24.50, 22.90, 23.80, 23.50],
    ("ST", "TURF", 1600, "4"): [24.50, 22.90, 23.80, 23.70],
    ("ST", "TURF", 1600, "5"): [24.55, 23.15, 23.85, 23.90],
}


def get_standard_time(venue: str, track: str, distance: int, cls: str) -> float | None:
    """Convenience lookup for standard time."""
    # Convert 'Class 3' or 'C3' to '3'
    import re
    m = re.search(r'\d', str(cls))
    cls_key = m.group(0) if m else str(cls).upper()
    if "GROUP" in str(cls).upper() or "GP" in str(cls).upper():
        cls_key = "GP"

    return STANDARD_TIMES.get((venue, track, distance, cls_key))


def get_z_score(time_sec: float, venue: str, track: str, distance: int, cls: str) -> float | None:
    """Calculate Z-score for a given time performance."""
    par = get_standard_time(venue, track, distance, cls)
    if not par:
        return None
    sd = STANDARD_DEVIATION_DEFAULTS.get(distance, 1.2)
    return (time_sec - par) / sd
