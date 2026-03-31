"""
EDI — Energy Distribution Index
Measures pacing consistency. Higher = more even pace throughout the race.

Formula:
    speeds = [section_dist / section_time for each section]
    EDI = 100 - (std(speeds) / mean(speeds)) * 100

Example (from technical doc):
    speeds = [13.58, 16.98, 16.68, 16.76, 17.08] m/s
    σ = 1.52, mean = 16.22
    EDI = 100 - (1.52 / 16.22) * 100 = 90.6
"""

from __future__ import annotations


import math


def calculate_edi(section_times: list[float], section_dist: float = 200.0) -> float:
    """
    Calculate Energy Distribution Index (EDI).

    Parameters
    ----------
    section_times : list[float]
        Split times per section (seconds).
    section_dist : float
        Distance of each section in metres (default 200m).
        Pass a list of floats if sections have different lengths.

    Returns
    -------
    float
        EDI value (0–100+). Higher = more consistent pacing.

    Ratings:
        > 90  : Excellent pace control
        85–90 : Good
        80–85 : Average
        < 80  : Erratic (front-running burnout or interference)
    """
    if isinstance(section_dist, (int, float)):
        speeds = [section_dist / t for t in section_times]
    else:
        # Variable section distances
        if len(section_dist) != len(section_times):
            raise ValueError("section_dist list must match section_times length.")
        speeds = [d / t for d, t in zip(section_dist, section_times)]

    n = len(speeds)
    if n < 2:
        raise ValueError("Need at least 2 sections to calculate EDI.")

    mean_v = sum(speeds) / n
    variance = sum((v - mean_v) ** 2 for v in speeds) / n
    std_v = math.sqrt(variance)

    return round(100 - (std_v / mean_v) * 100, 2)


def edi_rating_label(edi: float) -> str:
    if edi > 90:
        return "Excellent"
    elif edi >= 85:
        return "Good"
    elif edi >= 80:
        return "Average"
    else:
        return "Erratic"
