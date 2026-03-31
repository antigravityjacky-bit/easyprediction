"""
PA — Pace Adaptation Coefficient
Measures how well the horse coped with the actual pace of the race.

Formula:
    PA = 1 - |actual_early_speed - ideal_speed| / ideal_speed

Where ideal_speed is the benchmark early-pace speed for the distance.
"""

from __future__ import annotations


# Ideal front-600m average speeds by distance (m/s)
IDEAL_EARLY_SPEED = {
    1200: 15.8,
    1400: 15.5,
    1600: 15.2,
    1650: 15.1,
    1800: 14.9,
    2000: 14.7,
    2200: 14.5,
    2400: 14.3,
}


def get_ideal_early_speed(distance: int) -> float:
    """
    Return the ideal front-600m pace speed for a given distance.
    Interpolates linearly for distances not in the table.
    """
    if distance in IDEAL_EARLY_SPEED:
        return IDEAL_EARLY_SPEED[distance]

    # Linear interpolation between nearest defined distances
    keys = sorted(IDEAL_EARLY_SPEED.keys())
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= distance <= hi:
            ratio = (distance - lo) / (hi - lo)
            return IDEAL_EARLY_SPEED[lo] + ratio * (IDEAL_EARLY_SPEED[hi] - IDEAL_EARLY_SPEED[lo])

    # Extrapolate beyond table bounds
    if distance < keys[0]:
        return IDEAL_EARLY_SPEED[keys[0]]
    return IDEAL_EARLY_SPEED[keys[-1]]


def calculate_pa(
    section_times: list[float],
    distance: int,
    sections_for_early_pace: int = 3,
    section_dist: float = 200.0,
) -> float:
    """
    Calculate Pace Adaptation coefficient (PA).

    Parameters
    ----------
    section_times : list[float]
        Split times per section in seconds.
    distance : int
        Race distance in metres.
    sections_for_early_pace : int
        Number of leading sections used to measure early pace (default 3 = 600m).
    section_dist : float
        Distance per section in metres (default 200m).

    Returns
    -------
    float
        PA value between 0 and 1 (1.0 = perfect pace match).

    Ratings:
        >= 0.95 : Perfect pace
        0.90–0.95: Good pace
        < 0.90  : Pace mismatch (too fast or too slow early)
    """
    if len(section_times) < sections_for_early_pace:
        raise ValueError(
            f"Need at least {sections_for_early_pace} sections for early-pace calculation."
        )

    early_time = sum(section_times[:sections_for_early_pace])
    early_dist = section_dist * sections_for_early_pace
    actual_speed = early_dist / early_time

    ideal = get_ideal_early_speed(distance)
    pa = 1 - abs(actual_speed - ideal) / ideal
    return round(pa, 4)


def pa_rating_label(pa: float) -> str:
    if pa >= 0.95:
        return "Perfect Pace"
    elif pa >= 0.90:
        return "Good Pace"
    else:
        return "Pace Mismatch"
