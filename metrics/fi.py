"""
FI — Fatigue Index
Measures stamina retention. Ratio of final 400m speed vs peak 400m speed.

Formula:
    FI = (last_400m_avg_speed / fastest_400m_avg_speed) * 100

Extrapolation to longer distances:
    FI_new = FI_known * (known_distance / new_distance) ** 0.3

Example (from technical doc):
    Fastest 400m: 24.01s → 16.66 m/s
    Last 400m:    24.66s → 16.22 m/s
    FI = (16.22 / 16.66) * 100 = 97.4%
"""

from __future__ import annotations


SECTION_DIST_M = 400


def calculate_fi(section_times: list[float], section_dist: float = 200.0) -> float:
    """
    Calculate Fatigue Index (FI).

    Parameters
    ----------
    section_times : list[float]
        Split times per section (seconds). Assumes equal section_dist.
    section_dist : float
        Distance per section in metres (default 200m).

    Returns
    -------
    float
        FI as a percentage (0–100+). Higher = better stamina.

    Ratings:
        > 95  : Elite stamina
        90–95 : Excellent
        85–90 : Above average
        80–85 : Marginal
        < 80  : Stamina collapse
    """
    sections_per_400 = max(1, round(SECTION_DIST_M / section_dist))

    # Pair sections into 400m blocks
    n = len(section_times)
    blocks_400 = []
    i = 0
    while i + sections_per_400 <= n:
        block_time = sum(section_times[i:i + sections_per_400])
        block_dist = section_dist * sections_per_400
        blocks_400.append(block_dist / block_time)
        i += sections_per_400

    if len(blocks_400) < 2:
        raise ValueError("Need at least 2 complete 400m blocks to calculate FI.")

    fastest_speed = max(blocks_400)
    last_speed = blocks_400[-1]

    return round((last_speed / fastest_speed) * 100, 2)


def extrapolate_fi(fi_known: float, known_distance: int, new_distance: int) -> float:
    """
    Extrapolate FI to a new distance using the power-law decay formula.

    Parameters
    ----------
    fi_known : float
        FI measured at known_distance.
    known_distance : int
        Distance (metres) at which fi_known was measured.
    new_distance : int
        Target distance (metres) to project FI.

    Returns
    -------
    float
        Projected FI at new_distance.

    Example:
        fi_known=97.7, known=1800m, new=2200m → 92.0%
    """
    return round(fi_known * (known_distance / new_distance) ** 0.3, 2)


def fi_rating_label(fi: float) -> str:
    if fi > 95:
        return "Elite Stamina"
    elif fi >= 90:
        return "Excellent"
    elif fi >= 85:
        return "Above Average"
    elif fi >= 80:
        return "Marginal"
    else:
        return "Stamina Collapse"
