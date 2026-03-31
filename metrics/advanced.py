"""
Advanced Physics Metrics for Horse Racing Prediction

Extends the base 6 metrics (ASR, True Speed, FAP, EDI, FI, PA) with
15 additional physics-based indicators covering:
  A. Speed & Acceleration analysis
  B. Energy & Power estimation
  C. Track & Position physics
  D. Form & Fitness trends
"""

from __future__ import annotations

import math
from typing import Optional

from horseracing.constants.tracks import (
    TRACK_DATA, get_bend_ratio, get_rail_width,
)


# ── Physical Constants ───────────────────────────────────────────────────────

GRAVITY = 9.81                    # m/s²
AIR_DENSITY = 1.225               # kg/m³ at sea level
DRAG_COEFFICIENT = 0.9            # estimated Cd for horse+jockey
FRONTAL_AREA = 1.2                # m² estimated cross-section
ROLLING_FRICTION = 0.04           # turf rolling resistance coefficient
LB_TO_KG = 0.453592
OPTIMAL_REST_DAYS = 21            # peak freshness at ~3 weeks


# ═══════════════════════════════════════════════════════════════════════════════
# A. Speed & Acceleration Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def sectional_speeds(
    section_times: list[float],
    section_length: float = 200.0,
) -> list[float]:
    """
    Convert section split times to speeds (m/s).

    Parameters
    ----------
    section_times : list[float]
        Time in seconds for each section (typically 200m splits).
    section_length : float
        Length of each section in metres.

    Returns
    -------
    list[float] : speed in m/s for each section.
    """
    if not section_times:
        return []
    return [section_length / t if t > 0 else 0.0 for t in section_times]


def acceleration_profile(
    section_times: list[float],
    section_length: float = 200.0,
) -> list[float]:
    """
    Compute acceleration (m/s²) between adjacent sections.

    acceleration[i] = (v[i+1] - v[i]) / ((t[i] + t[i+1]) / 2)

    Positive = speeding up, negative = slowing down.
    Returns list of length len(section_times) - 1.
    """
    speeds = sectional_speeds(section_times, section_length)
    if len(speeds) < 2:
        return []
    accels = []
    for i in range(len(speeds) - 1):
        dt = (section_times[i] + section_times[i + 1]) / 2
        if dt > 0:
            accels.append((speeds[i + 1] - speeds[i]) / dt)
        else:
            accels.append(0.0)
    return accels


def peak_speed_section(section_times: list[float], section_length: float = 200.0) -> int:
    """
    Return the 0-based index of the section with highest speed.
    Returns -1 if no valid sections.
    """
    speeds = sectional_speeds(section_times, section_length)
    if not speeds:
        return -1
    return speeds.index(max(speeds))


def finishing_burst(section_times: list[float], section_length: float = 200.0) -> float:
    """
    Ratio of last-400m average speed to overall average speed.

    > 1.0 = horse accelerated at finish (strong kick)
    < 1.0 = horse faded at finish

    Uses last 2 sections (400m) if section_length=200.
    """
    speeds = sectional_speeds(section_times, section_length)
    if len(speeds) < 3:
        return 1.0

    overall_avg = sum(speeds) / len(speeds)
    if overall_avg <= 0:
        return 1.0

    # Last 2 sections = 400m
    n_finish = min(2, len(speeds))
    finish_avg = sum(speeds[-n_finish:]) / n_finish

    return round(finish_avg / overall_avg, 4)


def speed_decay_rate(section_times: list[float], section_length: float = 200.0) -> float:
    """
    Percentage change in average speed from first half to second half.

    Negative = horse slowed down (typical).
    Positive = horse sped up (unusual, strong finisher).

    Returns: (second_half_avg - first_half_avg) / first_half_avg × 100
    """
    speeds = sectional_speeds(section_times, section_length)
    if len(speeds) < 4:
        return 0.0

    mid = len(speeds) // 2
    first_half = sum(speeds[:mid]) / mid
    second_half = sum(speeds[mid:]) / len(speeds[mid:])

    if first_half <= 0:
        return 0.0

    return round((second_half - first_half) / first_half * 100, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# B. Energy & Power Estimation
# ═══════════════════════════════════════════════════════════════════════════════

def kinetic_energy_index(
    speed_ms: float,
    horse_weight_kg: int,
    carried_weight_lb: float,
) -> float:
    """
    Kinetic Energy Index = ½ × total_mass × v²

    total_mass = horse_weight_kg + carried_weight_lb × 0.4536

    Returns KE in joules. Higher = more energy needed to maintain speed
    at given load.
    """
    if speed_ms <= 0 or horse_weight_kg <= 0:
        return 0.0
    total_mass = horse_weight_kg + carried_weight_lb * LB_TO_KG
    return round(0.5 * total_mass * speed_ms ** 2, 1)


def weight_efficiency(finish_time: float, carried_weight_lb: float) -> float:
    """
    Seconds per pound carried. Lower = more efficient under weight.

    Returns finish_time / carried_weight_lb.
    """
    if carried_weight_lb <= 0 or finish_time <= 0:
        return 0.0
    return round(finish_time / carried_weight_lb, 4)


def power_output(
    total_mass_kg: float,
    speed_ms: float,
    gradient_deg: float = 0.0,
) -> float:
    """
    Estimated mechanical power output (Watts).

    P = v × (F_friction + F_gravity + F_drag)

    Where:
      F_friction = μ × m × g × cos(θ)
      F_gravity  = m × g × sin(θ)
      F_drag     = ½ × ρ × Cd × A × v²

    Parameters
    ----------
    total_mass_kg : float
        Horse weight + carried weight in kg.
    speed_ms : float
        Average speed in m/s.
    gradient_deg : float
        Track gradient in degrees (positive = uphill).
    """
    if total_mass_kg <= 0 or speed_ms <= 0:
        return 0.0

    theta = math.radians(gradient_deg)
    f_friction = ROLLING_FRICTION * total_mass_kg * GRAVITY * math.cos(theta)
    f_gravity = total_mass_kg * GRAVITY * math.sin(theta)
    f_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * FRONTAL_AREA * speed_ms ** 2

    total_force = f_friction + f_gravity + f_drag
    return round(speed_ms * total_force, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# C. Track & Position Physics
# ═══════════════════════════════════════════════════════════════════════════════

def turn_penalty(
    section_times: list[float],
    venue: str,
    distance: int,
    section_length: float = 200.0,
) -> float:
    """
    Speed loss in bend sections compared to straight sections.

    Uses the bend_ratio from tracks.py to estimate which sections are
    bends vs straights. Bend sections are assumed to be the early/middle
    segments (not the final straight).

    Returns: average speed in bends - average speed in straights (m/s).
    Negative = bends are slower (expected, especially at Happy Valley).
    """
    speeds = sectional_speeds(section_times, section_length)
    if len(speeds) < 3:
        return 0.0

    try:
        bend_ratio = get_bend_ratio(venue, distance)
    except (KeyError, ValueError):
        return 0.0

    if bend_ratio <= 0:
        return 0.0

    # Estimate number of bend sections
    total_sections = len(speeds)
    n_bend_sections = max(1, round(total_sections * bend_ratio))
    n_straight_sections = total_sections - n_bend_sections

    if n_straight_sections <= 0 or n_bend_sections <= 0:
        return 0.0

    # Assume last sections are straight (home straight), first sections
    # may include bends. For simplicity: sections [0..n_bend) are bend-heavy,
    # sections [n_bend..] are straight-heavy.
    # More accurate: skip first section (start) and last section (sprint finish)
    # to avoid noise from acceleration phases.
    if total_sections >= 5:
        # Exclude first and last sections (start acceleration / sprint finish)
        mid_speeds = speeds[1:-1]
        n_mid = len(mid_speeds)
        n_bend_mid = max(1, round(n_mid * bend_ratio))
        bend_speeds = mid_speeds[:n_bend_mid]
        straight_speeds = mid_speeds[n_bend_mid:]
    else:
        bend_speeds = speeds[:n_bend_sections]
        straight_speeds = speeds[n_bend_sections:]

    if not bend_speeds or not straight_speeds:
        return 0.0

    avg_bend = sum(bend_speeds) / len(bend_speeds)
    avg_straight = sum(straight_speeds) / len(straight_speeds)

    return round(avg_bend - avg_straight, 4)


def draw_bias_extra_distance(
    gate: int,
    venue: str,
    rail: str,
    distance: int,
) -> float:
    """
    Estimated extra running distance (metres) due to gate position.

    Horses drawn wider must cover more ground on bends.
    Extra distance ≈ gate_offset × 2π × bend_fraction × num_bends_traversed

    For a horse in gate N, assuming it runs ~1 lane width (1.8m) wider
    per gate number beyond gate 1.

    Returns extra distance in metres (always >= 0).
    """
    if gate <= 1:
        return 0.0

    try:
        data = TRACK_DATA[venue]
        dist_info = data["distances"].get(distance, {})
        bend_ratio = dist_info.get("bend_ratio", 0.3)
    except (KeyError, ValueError):
        bend_ratio = 0.3

    # Each gate adds ~1.8m of lane offset
    lane_offset = (gate - 1) * 1.8
    # Extra distance on bends: 2π × offset × bend_fraction
    # But we use the simpler model from true_speed.py: offset × bend_ratio
    # (no π — consistent with existing codebase convention)
    extra = lane_offset * bend_ratio
    return round(extra, 2)


def positioning_energy_cost(
    position_calls: list[int],
    section_times: list[float],
    section_length: float = 200.0,
) -> float:
    """
    Cumulative energy cost of changing running position.

    Each position change requires lateral movement + speed adjustment.
    Estimated as sum of |Δposition| weighted by section speed.

    Front-runners (low positions) pay less lateral cost but more
    wind resistance. Closers pay lateral cost when making their move.

    Returns: dimensionless cost score (higher = more positional work).
    """
    if len(position_calls) < 2 or len(section_times) < 2:
        return 0.0

    speeds = sectional_speeds(section_times, section_length)
    n = min(len(position_calls) - 1, len(speeds) - 1)

    cost = 0.0
    for i in range(n):
        pos_change = abs(position_calls[i + 1] - position_calls[i])
        if pos_change > 0 and i + 1 < len(speeds):
            # Cost proportional to position change × speed at that point
            # Normalized by typical speed (~16 m/s)
            cost += pos_change * (speeds[i + 1] / 16.0)

    return round(cost, 3)


def drafting_effect(position_calls: list[int]) -> float:
    """
    Estimated aerodynamic drag reduction from running behind other horses.

    Horses in positions >= 3 are typically sheltered from wind.
    Research suggests 6-12% drag reduction when drafting.

    Returns: average drag reduction factor (0.0 = no benefit, 0.12 = max).
    """
    if not position_calls:
        return 0.0

    total_reduction = 0.0
    for pos in position_calls:
        if pos >= 5:
            total_reduction += 0.12    # deep in field: maximum shelter
        elif pos >= 3:
            total_reduction += 0.08    # mid-pack: moderate shelter
        elif pos >= 2:
            total_reduction += 0.04    # stalking: slight shelter
        # pos == 1: leader, no drafting benefit

    return round(total_reduction / len(position_calls), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Form & Fitness Trends
# ═══════════════════════════════════════════════════════════════════════════════

def form_trend_index(recent_asr_values: list[float]) -> float:
    """
    Linear regression slope of recent ASR values.

    Positive slope = improving form.
    Negative slope = declining form.

    Uses simple least-squares fit: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    where x = race index (0, 1, 2, ...), y = ASR value.

    Parameters
    ----------
    recent_asr_values : list[float]
        ASR values ordered oldest → newest. Ideally 6 values.

    Returns
    -------
    float : slope (ASR units per race). Typical range: -5 to +5.
    """
    n = len(recent_asr_values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(recent_asr_values) / n

    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent_asr_values))
    den = sum((i - x_mean) ** 2 for i in range(n))

    if den == 0:
        return 0.0
    return round(num / den, 4)


def distance_aptitude(
    history: list[dict],
    target_distance: int,
    tolerance: int = 200,
) -> float:
    """
    Average true speed (m/s) at similar distances in history.

    Considers races within ±tolerance metres of target_distance.
    Returns 0.0 if no matching races.

    Parameters
    ----------
    history : list[dict]
        Race history dicts with keys: distance, finish_time, section_times.
    target_distance : int
        Target race distance in metres.
    tolerance : int
        Distance tolerance in metres.
    """
    speeds = []
    for race in history:
        dist = race.get("distance", 0)
        ft = race.get("finish_time", 0.0)
        if abs(dist - target_distance) <= tolerance and ft > 0 and dist > 0:
            speeds.append(dist / ft)

    if not speeds:
        return 0.0
    return round(sum(speeds) / len(speeds), 4)


def track_affinity(
    history: list[dict],
    target_venue: str,
) -> float:
    """
    Performance differential at target venue vs overall.

    Returns: avg_speed_at_venue - avg_speed_overall (m/s).
    Positive = horse performs better at this venue.
    """
    all_speeds = []
    venue_speeds = []

    venue_key = target_venue.upper()

    for race in history:
        ft = race.get("finish_time", 0.0)
        dist = race.get("distance", 0)
        if ft <= 0 or dist <= 0:
            continue
        speed = dist / ft
        all_speeds.append(speed)

        race_venue = race.get("venue", "")
        if not race_venue:
            vc = race.get("venue_code", "")
            race_venue = "SHA_TIN" if vc == "ST" else "HAPPY_VALLEY"

        if race_venue.upper() == venue_key:
            venue_speeds.append(speed)

    if not all_speeds or not venue_speeds:
        return 0.0

    overall_avg = sum(all_speeds) / len(all_speeds)
    venue_avg = sum(venue_speeds) / len(venue_speeds)
    return round(venue_avg - overall_avg, 4)


def freshness_factor(days_since_last_race: int) -> float:
    """
    Freshness score based on days since last race.

    Optimal rest period is ~21 days. Performance drops for both
    too-short rest (fatigue) and too-long rest (fitness loss).

    Model: Gaussian-like curve centered at OPTIMAL_REST_DAYS.
      score = exp(-((days - optimal)² / (2 × σ²)))

    σ = 14 days (empirical; covers 7-35 day window well).

    Returns: 0.0 to 1.0 (1.0 = optimal freshness).
    Returns 0.5 for unknown (days <= 0).
    """
    if days_since_last_race <= 0:
        return 0.5  # unknown

    sigma = 14.0
    z = (days_since_last_race - OPTIMAL_REST_DAYS) / sigma
    return round(math.exp(-0.5 * z * z), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: compute all advanced metrics for one entry
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_advanced(
    section_times: list[float],
    position_calls: list[float],
    finish_time: float,
    distance: int,
    venue: str,
    rail: str,
    gate: int,
    horse_weight_kg: int,
    carried_weight_lb: float,
    days_since_last: int = -1,
    recent_asr_values: Optional[list[float]] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    """
    Compute all advanced metrics for a single horse-in-race.

    Returns a flat dict with all 15+ metrics.
    """
    speeds = sectional_speeds(section_times)
    avg_speed = distance / finish_time if finish_time > 0 and distance > 0 else 0.0
    total_mass = horse_weight_kg + carried_weight_lb * LB_TO_KG if horse_weight_kg > 0 else 500.0

    # Gradient for power calculation
    gradient = 0.0
    if venue == "HAPPY_VALLEY":
        gradient = TRACK_DATA.get("HAPPY_VALLEY", {}).get("gradient", {}).get(
            "straight_incline_deg", 0.3
        )

    result = {
        # A. Speed & Acceleration
        "peak_speed_section": peak_speed_section(section_times),
        "peak_speed_ms": max(speeds) if speeds else 0.0,
        "finishing_burst": finishing_burst(section_times),
        "speed_decay_rate": speed_decay_rate(section_times),
        "max_acceleration": max(acceleration_profile(section_times), default=0.0),
        "min_acceleration": min(acceleration_profile(section_times), default=0.0),

        # B. Energy & Power
        "kinetic_energy_index": kinetic_energy_index(avg_speed, horse_weight_kg, carried_weight_lb),
        "weight_efficiency": weight_efficiency(finish_time, carried_weight_lb),
        "power_output_watts": power_output(total_mass, avg_speed, gradient),

        # C. Track & Position
        "turn_penalty_ms": turn_penalty(section_times, venue, distance),
        "draw_extra_distance_m": draw_bias_extra_distance(gate, venue, rail, distance),
        "positioning_cost": positioning_energy_cost(position_calls, section_times),
        "drafting_factor": drafting_effect(position_calls),

        # D. Form & Fitness
        "form_trend": form_trend_index(recent_asr_values or []),
        "freshness": freshness_factor(days_since_last),
    }

    # Distance aptitude and track affinity require history
    if history:
        result["distance_aptitude_ms"] = distance_aptitude(history, distance)
        result["track_affinity_ms"] = track_affinity(history, venue)
    else:
        result["distance_aptitude_ms"] = 0.0
        result["track_affinity_ms"] = 0.0

    return result
