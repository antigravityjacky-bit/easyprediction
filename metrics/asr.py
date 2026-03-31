"""
ASR — Absolute Speed Rating
Standardised speed ability comparable across races.

Formula:
    ASR = (distance / finish_time) * 100 * weight_factor * condition_factor

Weight factor:
    weight_factor = 1 + (130 - actual_weight) / 130 * 0.05

Example (from technical doc):
    distance=1800m, time=109.56s, weight=130lb, condition=GOOD_TO_YIELDING
    ASR = (1800/109.56) * 100 * 1.0 * 1.015 = 166.8
"""

from __future__ import annotations


from horseracing.constants.conditions import get_speed_factor

STANDARD_WEIGHT_LB = 130


def weight_factor(actual_weight_lb: float) -> float:
    """
    Calculate weight correction factor relative to 130lb standard.

    Parameters
    ----------
    actual_weight_lb : float
        Actual weight carried in pounds.

    Returns
    -------
    float
        Multiplicative correction factor (>1 if carrying less than standard).
    """
    return 1 + (STANDARD_WEIGHT_LB - actual_weight_lb) / STANDARD_WEIGHT_LB * 0.05


def calculate_asr(
    distance: int,
    finish_time: float,
    actual_weight_lb: float = STANDARD_WEIGHT_LB,
    condition: str = "GOOD",
) -> float:
    """
    Calculate Absolute Speed Rating (ASR).

    Parameters
    ----------
    distance : int
        Race distance in metres.
    finish_time : float
        Official finish time in seconds.
    actual_weight_lb : float
        Weight carried in pounds (default 130).
    condition : str
        Track condition canonical key or alias (e.g. "GOOD", "好地").

    Returns
    -------
    float
        ASR value. Higher = faster.

    Ratings:
        > 168  : Elite
        165-168: Excellent
        160-165: Above average
        155-160: Average
        < 155  : Below average
    """
    wf = weight_factor(actual_weight_lb)
    cf = get_speed_factor(condition)
    return (distance / finish_time) * 10 * wf * cf


def asr_rating_label(asr: float) -> str:
    if asr > 168:
        return "Elite"
    elif asr >= 165:
        return "Excellent"
    elif asr >= 160:
        return "Above Average"
    elif asr >= 155:
        return "Average"
    else:
        return "Below Average"
