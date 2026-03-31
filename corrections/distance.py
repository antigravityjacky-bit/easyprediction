"""
Distance extrapolation with fatigue correction.

Formula:
    time_new = time_base * (dist_new / dist_base) * fatigue_factor

Fatigue factor:
    FI_new = FI_known * (known_dist / new_dist) ** 0.3
    fatigue_factor = 1 + (1 - FI_new / 100) * 0.5
"""

from __future__ import annotations


from horseracing.metrics.fi import extrapolate_fi


def fatigue_factor(fi_at_base_distance: float, base_distance: int, new_distance: int) -> float:
    """
    Calculate fatigue correction factor for a new distance.

    Parameters
    ----------
    fi_at_base_distance : float
        FI (%) measured/averaged at base_distance.
    base_distance : int
        Distance (metres) at which FI was measured.
    new_distance : int
        Target distance (metres).

    Returns
    -------
    float
        Multiplicative factor applied to base time (>1 = slower at longer distance).
    """
    fi_new = extrapolate_fi(fi_at_base_distance, base_distance, new_distance)
    return round(1 + (1 - fi_new / 100) * 0.5, 6)


def extrapolate_time(
    base_time: float,
    base_distance: int,
    new_distance: int,
    fi_at_base: float,
) -> dict:
    """
    Extrapolate a finish time from one distance to another.

    Parameters
    ----------
    base_time : float
        Reference finish time in seconds.
    base_distance : int
        Distance (metres) of the reference race.
    new_distance : int
        Target distance (metres).
    fi_at_base : float
        FI (%) at base_distance.

    Returns
    -------
    dict with keys:
        fi_new, fatigue_factor, distance_ratio, extrapolated_time
    """
    dist_ratio = new_distance / base_distance
    fi_new = extrapolate_fi(fi_at_base, base_distance, new_distance)
    ff = fatigue_factor(fi_at_base, base_distance, new_distance)
    extrapolated = round(base_time * dist_ratio * ff, 3)

    return {
        "fi_new": fi_new,
        "fatigue_factor": ff,
        "distance_ratio": round(dist_ratio, 4),
        "extrapolated_time": extrapolated,
    }
