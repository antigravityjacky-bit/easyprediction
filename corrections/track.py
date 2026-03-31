"""
Track condition and venue conversion corrections.
"""

from __future__ import annotations


from horseracing.constants.conditions import (
    TRACK_CONDITION_FACTORS,
    get_condition_key,
    get_speed_factor,
)

# Venue-to-venue adjustment factors (relative to Sha Tin as baseline = 1.0)
# Derived from comparing equivalent races across venues.
VENUE_FACTORS = {
    "SHA_TIN":      1.000,
    "HAPPY_VALLEY": 1.006,   # tighter bends + gradient → marginally slower
}


def condition_conversion_factor(condition_from: str, condition_to: str) -> float:
    """
    Return the multiplicative time factor to convert a time from one
    track condition to another.

    factor > 1 → time gets longer (slower conditions)
    factor < 1 → time gets shorter (faster conditions)

    Example:
        GOOD_TO_YIELDING → GOOD
        factor = 1.000 / 1.015 = 0.985  (time decreases → faster)
    """
    sf_from = get_speed_factor(condition_from)
    sf_to = get_speed_factor(condition_to)
    # speed_factor is multiplied onto time (>1 = slower)
    return round(sf_to / sf_from, 6)


def venue_conversion_factor(venue_from: str, venue_to: str) -> float:
    """
    Return the multiplicative time factor to convert between venues.
    """
    return round(VENUE_FACTORS[venue_to] / VENUE_FACTORS[venue_from], 6)


def apply_track_corrections(
    base_time: float,
    condition_from: str,
    condition_to: str,
    venue_from: str,
    venue_to: str,
) -> float:
    """
    Apply condition and venue corrections to a base time.

    Parameters
    ----------
    base_time : float
        Reference finish time in seconds.
    condition_from : str
        Condition of the reference race.
    condition_to : str
        Condition of the target race.
    venue_from : str
        Venue of the reference race ("SHA_TIN" / "HAPPY_VALLEY").
    venue_to : str
        Venue of the target race.

    Returns
    -------
    float
        Adjusted time in seconds.
    """
    cf = condition_conversion_factor(condition_from, condition_to)
    vf = venue_conversion_factor(venue_from, venue_to)
    return round(base_time * cf * vf, 3)
