"""
Gate (draw) effect calculator.
Estimates time penalty and extra distance from wide barrier positions.
"""

from __future__ import annotations


from horseracing.constants.conditions import GATE_IMPACT_COEFFICIENTS
from horseracing.constants.tracks import get_bend_ratio, get_rail_width

ASSUMED_SPEED_MS = 16.0   # m/s — used for converting distance to time
LANE_WIDTH_M = 1.8        # metres per lane


def _estimate_avg_layers(gate: int) -> float:
    """
    Estimate average lane position (疊數) for a given barrier.
    Based on empirical observation that horses drift inward after the jump.
    """
    if gate <= 3:
        return float(gate)
    elif gate <= 6:
        return gate - 0.5
    elif gate <= 9:
        return gate - 1.0
    else:
        return gate - 1.5


def calculate_gate_effect(
    gate: int,
    venue: str,
    rail: str,
    distance: int,
) -> dict:
    """
    Calculate the time and distance penalty from a wide gate draw.

    Parameters
    ----------
    gate : int
        Barrier number (1–14).
    venue : str
        "SHA_TIN" or "HAPPY_VALLEY".
    rail : str
        Rail position: "A", "B", "C", or "C+3".
    distance : int
        Race distance in metres.

    Returns
    -------
    dict with keys:
        estimated_avg_layers, extra_distance_m, width_factor,
        time_loss_sec, gate_coefficient
    """
    avg_pos = _estimate_avg_layers(gate)
    bend_ratio = get_bend_ratio(venue, distance)
    track_width = get_rail_width(venue, rail)

    # Extra metres due to wide running on bends
    extra_dist = LANE_WIDTH_M * (avg_pos - 1) * bend_ratio

    # Narrower tracks amplify the effect
    width_factor = 1 + (30.5 - track_width) / 30.5

    # Convert extra distance to time loss
    time_loss = (extra_dist * width_factor) / ASSUMED_SPEED_MS

    # Coefficient-based simple estimate (for quick comparisons)
    coeff = GATE_IMPACT_COEFFICIENTS[venue][rail]
    coeff_time_loss = (gate - 1) * coeff

    return {
        "estimated_avg_layers": avg_pos,
        "extra_distance_m": round(extra_dist, 2),
        "width_factor": round(width_factor, 3),
        "time_loss_sec": round(time_loss, 3),
        "gate_coefficient": coeff,
        "coeff_time_loss_sec": round(coeff_time_loss, 3),
    }


def gate_delta(
    gate_ref: int,
    gate_target: int,
    venue_ref: str,
    venue_target: str,
    rail_ref: str,
    rail_target: str,
    distance_ref: int,
    distance_target: int,
) -> float:
    """
    Calculate the time difference (seconds) between two gate/venue/rail combinations.
    Negative = target is advantaged (less time lost).

    Used in the prediction engine when converting between reference race and target race.
    """
    ref = calculate_gate_effect(gate_ref, venue_ref, rail_ref, distance_ref)
    target = calculate_gate_effect(gate_target, venue_target, rail_target, distance_target)
    return round(target["time_loss_sec"] - ref["time_loss_sec"], 3)
