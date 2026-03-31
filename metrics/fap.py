"""
FAP — Final Acceleration Power
Measures finishing kick based on last 200m vs penultimate 200m.

Formula:
    a = (v_final² - v_penultimate²) / (2 * 200)
    FAP = a * 100

Example (from technical doc):
    Last 200m = 11.80s → v = 16.95 m/s
    Penultimate 200m = 12.02s → v = 16.64 m/s
    a = (16.95² - 16.64²) / 400 = 0.026 m/s²
    FAP = 2.6
"""

from __future__ import annotations


SECTION_DISTANCE_M = 200


def calculate_fap(section_times: list[float]) -> float:
    """
    Calculate Final Acceleration Power (FAP).

    Parameters
    ----------
    section_times : list[float]
        Split times per 200m section, ordered from start to finish.
        Must have at least 2 elements (penultimate and last sections).

    Returns
    -------
    float
        FAP value. Positive = accelerating into finish.

    Ratings:
        > 7  : Explosive (suits long straights)
        4–7  : Good
        2–4  : Average
        0–2  : Ordinary
        < 0  : Decelerating (fading)
    """
    if len(section_times) < 2:
        raise ValueError("Need at least 2 section times to calculate FAP.")

    t_last = section_times[-1]
    t_pen = section_times[-2]

    v_last = SECTION_DISTANCE_M / t_last
    v_pen = SECTION_DISTANCE_M / t_pen

    acceleration = (v_last ** 2 - v_pen ** 2) / (2 * SECTION_DISTANCE_M)
    return round(acceleration * 100, 2)


def fap_rating_label(fap: float) -> str:
    if fap > 7:
        return "Explosive"
    elif fap >= 4:
        return "Good"
    elif fap >= 2:
        return "Average"
    elif fap >= 0:
        return "Ordinary"
    else:
        return "Fading"
