"""
Weight (負磅) effect calculator.
Each pound above/below 130lb standard affects finish time.

Base rate: 0.042 seconds per pound per 1200m
Scales linearly with distance.
"""

from __future__ import annotations


STANDARD_WEIGHT_LB = 130
BASE_RATE_PER_LB_PER_1200M = 0.042


def calculate_weight_effect(
    actual_weight_lb: float,
    distance: int,
    standard_weight_lb: float = STANDARD_WEIGHT_LB,
) -> dict:
    """
    Calculate the time impact of carrying a non-standard weight.

    Parameters
    ----------
    actual_weight_lb : float
        Weight carried in the target race (pounds).
    distance : int
        Race distance in metres.
    standard_weight_lb : float
        Baseline weight for comparison (default 130lb).

    Returns
    -------
    dict with keys:
        weight_diff_lb, correction_factor, time_effect_sec

    Interpretation:
        time_effect_sec < 0  → lighter than standard → faster
        time_effect_sec > 0  → heavier than standard → slower
    """
    weight_diff = standard_weight_lb - actual_weight_lb   # positive = carrying less

    # Non-linear correction factor (used in ASR)
    correction_factor = 1 + weight_diff / standard_weight_lb * 0.05

    # Time effect: carrying less = negative seconds (faster)
    rate_per_lb = BASE_RATE_PER_LB_PER_1200M * (distance / 1200)
    time_effect = -weight_diff * rate_per_lb   # flip sign: less weight → negative = faster

    return {
        "weight_diff_lb": weight_diff,
        "correction_factor": round(correction_factor, 5),
        "time_effect_sec": round(time_effect, 3),
    }


def weight_delta(
    weight_ref: float,
    weight_target: float,
    distance: int,
) -> float:
    """
    Time difference (seconds) between two weight assignments at a given distance.
    Negative = target horse is lighter → faster.
    """
    ref = calculate_weight_effect(weight_ref, distance)
    target = calculate_weight_effect(weight_target, distance)
    return round(target["time_effect_sec"] - ref["time_effect_sec"], 3)
