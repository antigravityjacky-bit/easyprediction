"""
True Speed — accounts for extra distance run due to wide positions.

Extra distance formula:
    extra_distance = 1.8 * (avg_layers - 1) * bend_ratio

NOTE: Do NOT use π here. The factor 1.8 is the lane width in metres.
      Using 1.8π would overestimate by ~10x (common mistake — see doc §9.1).

True speed:
    actual_distance = race_distance + extra_distance
    true_speed = actual_distance / finish_time  (m/s)
"""

from __future__ import annotations


from horseracing.constants.tracks import get_bend_ratio

LANE_WIDTH_M = 1.8   # metres per lane (疊)


def extra_distance(avg_layers: float, bend_ratio: float) -> float:
    """
    Calculate extra metres run due to wide positioning.

    Parameters
    ----------
    avg_layers : float
        Average lane position (疊數) throughout the race. 1 = rail.
    bend_ratio : float
        Proportion of race distance spent on bends (0–1).

    Returns
    -------
    float
        Additional metres run compared to running on the rail.
    """
    return LANE_WIDTH_M * (avg_layers - 1) * bend_ratio


def calculate_true_speed(
    distance: int,
    finish_time: float,
    position_calls: list[int],
    venue: str,
) -> dict:
    """
    Calculate true speed accounting for wide running.

    Parameters
    ----------
    distance : int
        Official race distance in metres.
    finish_time : float
        Official finish time in seconds.
    position_calls : list[int]
        Lane positions per section (e.g. [11, 11, 10, 8, 3]).
    venue : str
        "SHA_TIN" or "HAPPY_VALLEY".

    Returns
    -------
    dict with keys:
        avg_layers, bend_ratio, extra_dist_m,
        actual_distance_m, true_speed_ms, official_speed_ms
    """
    avg_layers = sum(position_calls) / len(position_calls)
    br = get_bend_ratio(venue, distance)
    extra = extra_distance(avg_layers, br)
    actual_dist = distance + extra
    true_spd = actual_dist / finish_time
    official_spd = distance / finish_time

    return {
        "avg_layers": round(avg_layers, 2),
        "bend_ratio": br,
        "extra_dist_m": round(extra, 2),
        "actual_distance_m": round(actual_dist, 2),
        "true_speed_ms": round(true_spd, 4),
        "official_speed_ms": round(official_spd, 4),
    }
