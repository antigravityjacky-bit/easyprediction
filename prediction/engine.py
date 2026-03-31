"""
Prediction Engine
Full 7-step pipeline: baseline → distance → track → gate → weight → tactical → uncertainty.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Optional

from horseracing.profile.builder import HorseProfile
from horseracing.corrections.distance import extrapolate_time
from horseracing.corrections.track import apply_track_corrections
from horseracing.corrections.gate import gate_delta
from horseracing.corrections.weight import weight_delta
from horseracing.constants.tracks import get_straight_length

ASSUMED_SPEED_MS = 16.5   # m/s — used for straight-length tactical penalty
UNCERTAINTY_FLOOR = 0.45  # minimum σ: reflects unmodelled race-day variation


@dataclass
class RaceConditions:
    """Target race conditions."""
    venue: str                    # "SHA_TIN" / "HAPPY_VALLEY"
    rail: str                     # "A" / "B" / "C" / "C+3"
    distance: int                 # metres
    condition: str                # canonical condition key
    gate: int
    draw_weight_lb: float
    expected_pace: str = "NORMAL" # "FAST" / "NORMAL" / "SLOW"


@dataclass
class PredictionResult:
    horse_id: str
    horse_name: str
    predicted_time: float          # seconds
    uncertainty_sec: float         # 1-sigma
    ci_95_lower: float
    ci_95_upper: float
    step_log: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    key_assumptions: list[str] = field(default_factory=list)


def _straight_length_penalty(
    fap: float,
    venue_ref: str,
    rail_ref: str,
    venue_target: str,
    rail_target: str,
    base_time: float,
) -> float:
    """
    Penalty/bonus (seconds) for the difference in straight length between
    the reference race venue and the target race venue.

    Closers (high FAP) suffer when the straight is shorter.
    Formula scales by the proportion of straight lost.
    """
    sl_ref = get_straight_length(venue_ref, rail_ref)
    sl_target = get_straight_length(venue_target, rail_target)

    if sl_ref == sl_target:
        return 0.0

    # Impact coefficient: 0.3s per 100m of straight lost × FAP scaling
    straight_diff_ratio = (sl_ref - sl_target) / sl_ref
    fap_scale = max(0.5, min(fap / 5.0, 2.0))   # cap between 0.5–2.0
    penalty = straight_diff_ratio * 0.3 * fap_scale

    return round(penalty, 3)


def _pace_adjustment(pa: float, expected_pace: str, racing_style: str = "stalker") -> float:
    """
    Adjust time based on expected race pace × running style interaction.
    Front-runners are hurt most in FAST pace; closers benefit.
    Front-runners benefit most in SLOW pace; closers are hurt.
    PA modulates the magnitude (high PA = comfortable in any pace).
    """
    if expected_pace == "NORMAL":
        return 0.0

    pa_mod = 1 - pa   # 0 = perfectly adapted, 1 = poorly adapted

    if expected_pace == "FAST":
        if racing_style == "front-runner":
            return round(0.25 + pa_mod * 0.30, 3)    # +0.25–0.55s: front-runners tire
        elif racing_style == "closer":
            return round(-0.15 - pa_mod * 0.10, 3)   # −0.15–0.25s: closers benefit
        else:  # stalker
            return round(0.05 + pa_mod * 0.15, 3)    # small penalty

    elif expected_pace == "SLOW":
        if racing_style == "front-runner":
            return round(-0.20 - pa_mod * 0.10, 3)   # −0.20–0.30s: front-runners coast
        elif racing_style == "closer":
            return round(0.15 + pa_mod * 0.10, 3)    # +0.15–0.25s: closers lose run-in
        else:  # stalker
            return round(-0.05 + pa_mod * 0.05, 3)   # near-neutral

    return 0.0


def _uncertainty_estimate(
    profile: HorseProfile,
    is_first_time_at_venue: bool,
    is_distance_extrapolated: bool,
    distance_range_unc: float = 0.0,
) -> float:
    """
    Estimate 1-sigma uncertainty in seconds from multiple sources.
    Combined in quadrature (independent sources), floored at UNCERTAINTY_FLOOR
    to prevent overconfident win probabilities when a horse's metrics are tight.
    """
    # Source 1: historical variance in true speed
    hist_std = profile.true_speed.std if profile.true_speed else 0.2
    hist_time_std = hist_std * (
        profile.recent_races[-1].distance / ASSUMED_SPEED_MS ** 2
        if profile.recent_races else 1.0
    )

    # Source 2: first time at venue
    venue_uncertainty = 0.5 if is_first_time_at_venue else 0.1

    # Source 3: distance extrapolation uncertainty
    dist_uncertainty = 0.4 if is_distance_extrapolated else 0.1

    # Source 4: distance outside preferred range (caller-supplied)
    combined = (
        hist_time_std ** 2
        + venue_uncertainty ** 2
        + dist_uncertainty ** 2
        + distance_range_unc ** 2
    ) ** 0.5

    # Floor: even a perfectly consistent horse has unmodelled race-day variation
    return round(max(combined, UNCERTAINTY_FLOOR), 3)


def predict(
    profile: HorseProfile,
    target: RaceConditions,
) -> PredictionResult:
    """
    Full 7-step prediction pipeline for a single horse.

    Parameters
    ----------
    profile : HorseProfile
        Rolling 3-race average profile.
    target : RaceConditions
        Conditions for the race being predicted.

    Returns
    -------
    PredictionResult
    """
    log = []
    risks = []
    assumptions = []

    if not profile.recent_races:
        raise ValueError(f"No race history for horse {profile.horse_id}")

    # ── Step 1: Baseline from most recent race ──────────────────────────────
    ref = profile.recent_races[-1]
    base_time = ref.corrected_time
    base_speed = profile.true_speed.mean if profile.true_speed else ASSUMED_SPEED_MS
    avg_fi = profile.fi.mean if profile.fi else 95.0
    avg_fap = profile.fap.mean if profile.fap else 3.0
    avg_pa = profile.pa if profile.pa else 0.95
    racing_style = profile.racing_style or "stalker"
    log.append(f"[1] Baseline: {base_time:.2f}s @ {ref.distance}m {ref.venue} {ref.condition}")

    # ── Step 1b: Trend momentum ──────────────────────────────────────────────
    t = base_time
    if profile.true_speed and len(profile.true_speed.values) >= 2:
        delta_v = profile.true_speed.values[-1] - profile.true_speed.values[0]
        # Δt ≈ −Δv × (distance / v_mean²), damped at 40% to be conservative
        raw_adj = -delta_v * (ref.distance / base_speed ** 2) * 0.4
        trend_adj = max(-0.40, min(0.40, raw_adj))   # cap at ±0.4s
        t += trend_adj
        if abs(trend_adj) >= 0.01:
            log.append(
                f"[1b] Trend ({profile.true_speed.trend}, Δv={delta_v:+.4f}m/s): "
                f"Δ{trend_adj:+.3f}s → {t:.3f}s"
            )
    else:
        t = base_time

    # ── Step 2: Distance extrapolation ──────────────────────────────────────
    is_dist_extrap = target.distance != ref.distance
    dist_result = extrapolate_time(
        base_time=t,
        base_distance=ref.distance,
        new_distance=target.distance,
        fi_at_base=avg_fi,
    )
    t = dist_result["extrapolated_time"]
    log.append(
        f"[2] Distance {ref.distance}m→{target.distance}m: "
        f"FI_new={dist_result['fi_new']}%, ff={dist_result['fatigue_factor']:.4f} → {t:.3f}s"
    )

    # ── Step 3: Track condition + venue correction ───────────────────────────
    is_first_venue = not any(r.venue == target.venue for r in profile.recent_races)
    t = apply_track_corrections(
        base_time=t,
        condition_from=ref.condition,
        condition_to=target.condition,
        venue_from=ref.venue,
        venue_to=target.venue,
    )
    log.append(f"[3] Track correction ({ref.condition}→{target.condition}, {ref.venue}→{target.venue}): {t:.3f}s")
    if is_first_venue:
        risks.append(f"First time at {target.venue} (+0.5s uncertainty)")

    # ── Step 4: Gate correction ──────────────────────────────────────────────
    gate_adj = gate_delta(
        gate_ref=ref.gate,
        gate_target=target.gate,
        venue_ref=ref.venue,
        venue_target=target.venue,
        rail_ref=ref.rail,
        rail_target=target.rail,
        distance_ref=ref.distance,
        distance_target=target.distance,
    )
    t += gate_adj
    log.append(f"[4] Gate {ref.gate}→{target.gate}: Δ{gate_adj:+.3f}s → {t:.3f}s")
    if target.gate >= 10:
        risks.append(f"Wide gate {target.gate} — high risk of being pushed wider")
    else:
        assumptions.append(f"Gate {target.gate} holds position cleanly")

    # ── Step 5: Weight correction ────────────────────────────────────────────
    weight_adj = weight_delta(
        weight_ref=ref.draw_weight_lb,
        weight_target=target.draw_weight_lb,
        distance=target.distance,
    )
    t += weight_adj
    log.append(
        f"[5] Weight {ref.draw_weight_lb}lb→{target.draw_weight_lb}lb: "
        f"Δ{weight_adj:+.3f}s → {t:.3f}s"
    )

    # ── Step 6: Tactical adjustments ────────────────────────────────────────
    # 6a. Straight length penalty (affects closers most)
    sl_penalty = _straight_length_penalty(
        fap=avg_fap,
        venue_ref=ref.venue,
        rail_ref=ref.rail,
        venue_target=target.venue,
        rail_target=target.rail,
        base_time=t,
    )
    t += sl_penalty
    if sl_penalty != 0:
        log.append(f"[6a] Straight length penalty: Δ{sl_penalty:+.3f}s → {t:.3f}s")

    # 6b. Pace adaptation (style-aware)
    pace_adj = _pace_adjustment(avg_pa, target.expected_pace, racing_style)
    t += pace_adj
    if pace_adj != 0:
        log.append(
            f"[6b] Pace adjustment ({target.expected_pace}, {racing_style}): "
            f"Δ{pace_adj:+.3f}s → {t:.3f}s"
        )
    else:
        log.append(f"[6b] Pace: {target.expected_pace} — no adjustment")
        assumptions.append("Normal race pace assumed")

    # ── Step 7: Uncertainty ──────────────────────────────────────────────────
    distance_range_unc = 0.0
    if profile.preferred_distance_range:
        lo, hi = profile.preferred_distance_range
        if target.distance < lo * 0.87 or target.distance > hi * 1.13:
            distance_range_unc = 0.35
            risks.append(f"距離 {target.distance}m 遠超偏好範圍 {lo}–{hi}m (+0.35s σ)")
        elif target.distance < lo * 0.94 or target.distance > hi * 1.06:
            distance_range_unc = 0.15
            risks.append(f"距離 {target.distance}m 略超偏好範圍 {lo}–{hi}m (+0.15s σ)")
    sigma = _uncertainty_estimate(profile, is_first_venue, is_dist_extrap, distance_range_unc)
    ci_lower = round(t - 1.96 * sigma, 3)
    ci_upper = round(t + 1.96 * sigma, 3)
    log.append(f"[7] σ={sigma:.3f}s → 95% CI [{ci_lower:.2f}s, {ci_upper:.2f}s]")

    return PredictionResult(
        horse_id=profile.horse_id,
        horse_name=profile.horse_name,
        predicted_time=round(t, 3),
        uncertainty_sec=sigma,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        step_log=log,
        risk_factors=risks,
        key_assumptions=assumptions,
    )
