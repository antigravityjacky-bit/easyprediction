"""
Report generator — formats prediction results for display.
"""

from __future__ import annotations


from horseracing.prediction.engine import PredictionResult
from horseracing.prediction.monte_carlo import SimulationResult, HorseEntry


def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:05.2f}"


def generate_report(
    predictions: list[PredictionResult],
    sim: SimulationResult,
    entries: list[HorseEntry],
    race_title: str = "Race Prediction Report",
) -> str:
    """
    Generate a human-readable prediction report.

    Parameters
    ----------
    predictions : list[PredictionResult]
        Engine output for each horse (one per horse).
    sim : SimulationResult
        Monte Carlo simulation output.
    entries : list[HorseEntry]
        Horse entries with metadata.
    race_title : str
        Title for the report header.

    Returns
    -------
    str
        Formatted report string.
    """
    # Build lookup maps
    pred_map = {p.horse_id: p for p in predictions}
    entry_map = {e.horse_id: e for e in entries}

    # Sort by win probability descending
    sorted_horses = sorted(sim.win_probs.items(), key=lambda x: -x[1])

    lines = [
        "=" * 70,
        f"  {race_title}",
        f"  Monte Carlo: {sim.n_simulations:,} simulations | Field: {sim.field_size} horses",
        "=" * 70,
        "",
        f"{'Rank':<5} {'Horse':<20} {'Gate':<5} {'Pred Time':<12} {'95% CI':<22} {'Win%':<8} {'Place%':<8}",
        "-" * 85,
    ]

    for rank, (horse_id, win_prob) in enumerate(sorted_horses, 1):
        pred = pred_map.get(horse_id)
        entry = entry_map.get(horse_id)
        if pred is None or entry is None:
            continue

        name = pred.horse_name[:19]
        pt = _fmt_time(pred.predicted_time)
        ci = f"[{_fmt_time(pred.ci_95_lower)} – {_fmt_time(pred.ci_95_upper)}]"
        win_pct = f"{win_prob * 100:.1f}%"
        place_pct = f"{sim.place_probs[horse_id] * 100:.1f}%"

        lines.append(
            f"{rank:<5} {name:<20} {entry.gate:<5} {pt:<12} {ci:<22} {win_pct:<8} {place_pct:<8}"
        )

    lines += ["", "── Risk Factors & Assumptions ──────────────────────────────────────", ""]

    for horse_id, _ in sorted_horses:
        pred = pred_map.get(horse_id)
        if pred is None:
            continue
        if pred.risk_factors or pred.key_assumptions:
            lines.append(f"  {pred.horse_name}:")
            for r in pred.risk_factors:
                lines.append(f"    ⚠  {r}")
            for a in pred.key_assumptions:
                lines.append(f"    ✓  {a}")
            lines.append("")

    lines += [
        "── Pace Interaction Summary ────────────────────────────────────────",
        "",
    ]
    front_runners = [e.horse_name for e in entries if e.racing_style == "front-runner"]
    closers = [e.horse_name for e in entries if e.racing_style == "closer"]
    stalkers = [e.horse_name for e in entries if e.racing_style == "stalker"]

    if front_runners:
        lines.append(f"  Front-runners  : {', '.join(front_runners)}")
    if stalkers:
        lines.append(f"  Stalkers       : {', '.join(stalkers)}")
    if closers:
        lines.append(f"  Closers        : {', '.join(closers)}")

    pace_pressure_desc = "Hot" if len(front_runners) >= 3 else "Moderate" if len(front_runners) == 2 else "Slow"
    lines.append(f"  Expected Pace  : {pace_pressure_desc}")
    lines += ["", "=" * 70]

    return "\n".join(lines)
