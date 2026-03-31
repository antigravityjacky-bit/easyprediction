"""
Enhanced Counter-Trend Horse Identification Engine

Identifies horses whose physical data suggests they are better than
recent results indicate — key "dark horse" candidates.

Five detection dimensions:
  1. Physical uptrend (improving metrics despite poor finishes)
  2. Distance/surface match (first time at optimal conditions)
  3. Weight relief (dropped weight, maintained ability)
  4. Conditions advantage (draw, track, weather favor this horse)
  5. Bad luck correction (recent interference/gate penalties)

Each sub-score is 0-100. Composite is a weighted sum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from horseracing.metrics.advanced import (
    finishing_burst,
    speed_decay_rate,
    form_trend_index,
    distance_aptitude,
    track_affinity,
    freshness_factor,
    draw_bias_extra_distance,
    sectional_speeds,
)
from horseracing.scraper.stewards import estimate_total_loss


# ── Weights for composite score ──────────────────────────────────────────────

WEIGHTS = {
    "physical_uptrend": 0.30,
    "distance_match": 0.20,
    "weight_relief": 0.20,
    "conditions_advantage": 0.15,
    "bad_luck_correction": 0.15,
}


@dataclass
class CounterTrendResult:
    """Result of counter-trend analysis for one horse."""
    horse_code: str
    physical_uptrend_score: float = 0.0
    distance_match_score: float = 0.0
    weight_relief_score: float = 0.0
    conditions_advantage_score: float = 0.0
    bad_luck_correction_score: float = 0.0
    composite: float = 0.0
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "horse_code": self.horse_code,
            "physical_uptrend": self.physical_uptrend_score,
            "distance_match": self.distance_match_score,
            "weight_relief": self.weight_relief_score,
            "conditions_advantage": self.conditions_advantage_score,
            "bad_luck_correction": self.bad_luck_correction_score,
            "composite": self.composite,
            "signals": self.signals,
        }


# ── Sub-score calculators ────────────────────────────────────────────────────

def _physical_uptrend_score(history: list[dict]) -> tuple[float, list[str]]:
    """
    Detect improving physical data despite poor finishes.

    Looks at:
    - Finishing burst trend (improving last-400m kick)
    - Speed decay improvement (less fading)
    - Sectional speed improvement in final sections
    - Despite finish_position > 5 in recent races (unlucky/blocked)

    Returns (score 0-100, signals).
    """
    signals = []
    if len(history) < 2:
        return 0.0, signals

    recent = history[-3:] if len(history) >= 3 else history

    # Check finishing burst trend
    bursts = []
    for race in recent:
        st = race.get("section_times", [])
        if st and len(st) >= 3:
            bursts.append(finishing_burst(st))

    burst_improving = False
    if len(bursts) >= 2 and bursts[-1] > bursts[0]:
        burst_improving = True
        signals.append(f"末段爆發力提升 {bursts[0]:.3f}→{bursts[-1]:.3f}")

    # Check speed decay trend (less negative = better)
    decays = []
    for race in recent:
        st = race.get("section_times", [])
        if st and len(st) >= 4:
            decays.append(speed_decay_rate(st))

    decay_improving = False
    if len(decays) >= 2 and decays[-1] > decays[0]:
        decay_improving = True
        signals.append(f"速度衰減改善 {decays[0]:.1f}%→{decays[-1]:.1f}%")

    # Check if finishes were poor despite physical improvement
    recent_positions = [r.get("finish_position", 0) for r in recent if r.get("finish_position")]
    poor_finishes = sum(1 for p in recent_positions if p > 5)
    has_poor_finishes = poor_finishes >= len(recent_positions) * 0.5

    # Score calculation
    score = 0.0
    if burst_improving:
        score += 35
    if decay_improving:
        score += 25
    if has_poor_finishes and (burst_improving or decay_improving):
        score += 30  # Physical data says better than results show
        signals.append(f"近{len(recent)}仗名次偏差但物理數據上升")
    elif burst_improving and decay_improving:
        score += 10

    return min(score, 100.0), signals


def _distance_match_score(
    history: list[dict],
    target_distance: int,
) -> tuple[float, list[str]]:
    """
    Detect horses trying a more suitable distance.

    Uses distance_aptitude to find best historical distance,
    then checks if target_distance is closer to optimal.
    """
    signals = []
    if not history:
        return 0.0, signals

    # Find distances the horse has raced at
    dist_speeds: dict[int, list[float]] = {}
    for race in history:
        dist = race.get("distance", 0)
        ft = race.get("finish_time", 0.0)
        if dist > 0 and ft > 0:
            dist_speeds.setdefault(dist, []).append(dist / ft)

    if not dist_speeds:
        return 0.0, signals

    # Find best distance (highest avg speed)
    best_dist = max(dist_speeds, key=lambda d: sum(dist_speeds[d]) / len(dist_speeds[d]))
    best_speed = sum(dist_speeds[best_dist]) / len(dist_speeds[best_dist])

    # Speed at target distance (if available)
    apt = distance_aptitude(history, target_distance)

    # Has the horse tried this distance before?
    tried_target = any(abs(r.get("distance", 0) - target_distance) <= 100 for r in history)

    score = 0.0

    if not tried_target:
        # First time at this distance — check if physical profile suggests suitability
        # If target is closer to best_dist than recent distances, it's a good sign
        recent_dists = [r.get("distance", 0) for r in history[-3:] if r.get("distance")]
        if recent_dists:
            recent_avg_dist = sum(recent_dists) / len(recent_dists)
            # Is target closer to best than recent average?
            recent_diff = abs(recent_avg_dist - best_dist)
            target_diff = abs(target_distance - best_dist)
            if target_diff < recent_diff:
                score += 60
                signals.append(f"首次嘗試{target_distance}m，更接近最佳距離{best_dist}m")
            else:
                score += 20
                signals.append(f"首次嘗試{target_distance}m")
    elif apt > 0:
        # Has tried — check if apt speed is near best
        ratio = apt / best_speed if best_speed > 0 else 0
        if ratio >= 0.98:
            score += 50
            signals.append(f"在{target_distance}m表現接近最佳（{ratio:.3f}）")
        elif ratio >= 0.95:
            score += 30

    return min(score, 100.0), signals


def _weight_relief_score(
    history: list[dict],
    target_weight_lb: float,
) -> tuple[float, list[str]]:
    """
    Detect horses whose weight has dropped significantly but ability remains.

    Key indicator: weight dropped >= 3lb but sectional speeds stable/improving.
    """
    signals = []
    if not history or target_weight_lb <= 0:
        return 0.0, signals

    # Recent carried weights
    recent_weights = [
        r.get("draw_weight_lb", 0.0) for r in history[-3:]
        if r.get("draw_weight_lb", 0) > 0
    ]
    if not recent_weights:
        return 0.0, signals

    avg_recent_weight = sum(recent_weights) / len(recent_weights)
    weight_drop = avg_recent_weight - target_weight_lb

    if weight_drop < 2.0:
        return 0.0, signals

    # Check if ability is maintained (sectional speeds not declining)
    speeds_trend = []
    for race in history[-3:]:
        st = race.get("section_times", [])
        if st:
            sp = sectional_speeds(st)
            if sp:
                speeds_trend.append(sum(sp) / len(sp))

    ability_maintained = True
    if len(speeds_trend) >= 2:
        if speeds_trend[-1] < speeds_trend[0] * 0.97:
            ability_maintained = False

    score = 0.0
    if weight_drop >= 5 and ability_maintained:
        score = 80
        signals.append(f"負磅大降{weight_drop:.0f}lb（{avg_recent_weight:.0f}→{target_weight_lb:.0f}），能力未減")
    elif weight_drop >= 3 and ability_maintained:
        score = 50
        signals.append(f"負磅下調{weight_drop:.0f}lb，速度穩定")
    elif weight_drop >= 3:
        score = 25
        signals.append(f"負磅下調{weight_drop:.0f}lb")

    return min(score, 100.0), signals


def _conditions_advantage_score(
    history: list[dict],
    target_venue: str,
    target_gate: int,
    target_rail: str,
    target_distance: int,
    target_condition: str,
) -> tuple[float, list[str]]:
    """
    Detect horses with favorable race-day conditions.

    Checks:
    - Gate improvement vs recent races
    - Track affinity (does well at this venue)
    - Condition suitability (running style vs track condition)
    """
    signals = []
    score = 0.0

    if not history:
        return 0.0, signals

    # Gate improvement
    recent_gates = [r.get("gate", 0) for r in history[-3:] if r.get("gate", 0) > 0]
    if recent_gates and target_gate > 0:
        avg_recent_gate = sum(recent_gates) / len(recent_gates)
        # Lower gate is generally better (especially at HV)
        gate_improvement = avg_recent_gate - target_gate
        if gate_improvement >= 4:
            score += 30
            signals.append(f"檔位大幅改善（平均{avg_recent_gate:.0f}→{target_gate}）")
        elif gate_improvement >= 2:
            score += 15
            signals.append(f"檔位改善（{avg_recent_gate:.0f}→{target_gate}）")

    # Track affinity
    aff = track_affinity(history, target_venue)
    if aff > 0.2:
        score += 25
        signals.append(f"場地親和度高（+{aff:.3f} m/s）")
    elif aff > 0.1:
        score += 10

    # Draw bias at venue — check if gate is favorable
    try:
        extra_dist = draw_bias_extra_distance(target_gate, target_venue, target_rail, target_distance)
        if extra_dist < 1.0:
            score += 15
            signals.append(f"檔位有利（額外距離僅{extra_dist:.1f}m）")
    except Exception:
        pass

    # Condition suitability — closers benefit from softer ground
    # (approximated by checking if horse tends to come from behind)
    recent_positions = []
    for race in history[-3:]:
        pc = race.get("position_calls", [])
        if pc:
            recent_positions.append(pc[0] if pc else 0)  # early position

    if recent_positions:
        avg_early_pos = sum(recent_positions) / len(recent_positions)
        is_closer = avg_early_pos > 7
        soft_conditions = target_condition in ("YIELDING", "SLOW", "GOOD_TO_YIELDING")

        if is_closer and soft_conditions:
            score += 20
            signals.append("後上馬在軟地上更具優勢")

    return min(score, 100.0), signals


def _bad_luck_correction_score(history: list[dict]) -> tuple[float, list[str]]:
    """
    Detect horses with recent bad luck (interference, bad draws, slow starts).

    Checks stewards notes and gate penalties in recent races.
    """
    signals = []
    if not history:
        return 0.0, signals

    total_interference = 0.0
    bad_gate_races = 0

    for race in history[-3:]:
        # Stewards interference
        note = race.get("stewards_note", "")
        if note:
            loss = estimate_total_loss(note)
            total_interference += loss

        # Bad gate (>= 10 at ST, >= 8 at HV)
        gate = race.get("gate", 0)
        venue_code = race.get("venue_code", "")
        venue = race.get("venue", "")
        if venue_code == "HV" or "HAPPY" in venue.upper():
            if gate >= 8:
                bad_gate_races += 1
        elif gate >= 10:
            bad_gate_races += 1

    score = 0.0

    if total_interference >= 0.5:
        score += 40
        signals.append(f"近仗累計受干擾損失{total_interference:.2f}秒")
    elif total_interference > 0:
        score += 20
        signals.append(f"近仗有輕微干擾（{total_interference:.2f}秒）")

    if bad_gate_races >= 2:
        score += 35
        signals.append(f"近{bad_gate_races}仗抽到不利檔位")
    elif bad_gate_races == 1:
        score += 15

    # Check if poor position calls suggest being blocked
    blocked_count = 0
    for race in history[-3:]:
        pc = race.get("position_calls", [])
        fp = race.get("finish_position", 0)
        if len(pc) >= 2 and fp > 0:
            # Was stuck in same position through middle of race then dropped
            mid_positions = pc[1:-1] if len(pc) > 2 else pc
            if mid_positions and max(mid_positions) - min(mid_positions) <= 1 and fp > pc[-1]:
                # Couldn't improve position, finished worse than last call
                pass
            elif len(pc) >= 3 and pc[-1] > pc[-2] + 2:
                # Dropped sharply at end — possible interference
                blocked_count += 1

    if blocked_count >= 2:
        score += 25
        signals.append("近仗走位受阻跡象")

    return min(score, 100.0), signals


# ── Main API ─────────────────────────────────────────────────────────────────

def score_counter_trend(
    horse_code: str,
    history: list[dict],
    target_distance: int,
    target_venue: str,
    target_gate: int = 0,
    target_weight_lb: float = 0.0,
    target_rail: str = "A",
    target_condition: str = "GOOD",
) -> CounterTrendResult:
    """
    Run all 5 counter-trend detectors for one horse.

    Parameters
    ----------
    horse_code : str
        Horse identifier.
    history : list[dict]
        Past race dicts (oldest first). Each should contain:
        date, distance, finish_time, finish_position, section_times,
        position_calls, draw_weight_lb, gate, venue/venue_code, stewards_note.
    target_distance : int
        Distance of the upcoming race.
    target_venue : str
        "SHA_TIN" or "HAPPY_VALLEY".
    target_gate : int
        Gate draw for upcoming race.
    target_weight_lb : float
        Carried weight for upcoming race.
    target_rail : str
        Rail position.
    target_condition : str
        Track condition.

    Returns
    -------
    CounterTrendResult with sub-scores and composite.
    """
    result = CounterTrendResult(horse_code=horse_code)

    # 1. Physical uptrend
    s1, sig1 = _physical_uptrend_score(history)
    result.physical_uptrend_score = s1
    result.signals.extend(sig1)

    # 2. Distance match
    s2, sig2 = _distance_match_score(history, target_distance)
    result.distance_match_score = s2
    result.signals.extend(sig2)

    # 3. Weight relief
    s3, sig3 = _weight_relief_score(history, target_weight_lb)
    result.weight_relief_score = s3
    result.signals.extend(sig3)

    # 4. Conditions advantage
    s4, sig4 = _conditions_advantage_score(
        history, target_venue, target_gate, target_rail,
        target_distance, target_condition,
    )
    result.conditions_advantage_score = s4
    result.signals.extend(sig4)

    # 5. Bad luck correction
    s5, sig5 = _bad_luck_correction_score(history)
    result.bad_luck_correction_score = s5
    result.signals.extend(sig5)

    # Composite weighted score
    result.composite = round(
        s1 * WEIGHTS["physical_uptrend"]
        + s2 * WEIGHTS["distance_match"]
        + s3 * WEIGHTS["weight_relief"]
        + s4 * WEIGHTS["conditions_advantage"]
        + s5 * WEIGHTS["bad_luck_correction"],
        1,
    )

    return result
