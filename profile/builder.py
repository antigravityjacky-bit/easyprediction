"""
Horse Profile Builder
Computes physical metrics for a race entry and maintains a rolling
3-race average profile per horse.
"""

from __future__ import annotations


import math
from dataclasses import dataclass, field
from typing import Optional

from horseracing.metrics.asr import calculate_asr
from horseracing.metrics.true_speed import calculate_true_speed
from horseracing.metrics.fap import calculate_fap
from horseracing.metrics.edi import calculate_edi
from horseracing.metrics.fi import calculate_fi
from horseracing.metrics.pa import calculate_pa

MAX_ROLLING_RACES = 3


@dataclass
class RaceEntry:
    """Raw data for a single horse in a single race."""
    race_id: str
    horse_id: str
    date: str                        # ISO format "YYYY-MM-DD"
    venue: str                       # "SHA_TIN" / "HAPPY_VALLEY"
    rail: str                        # "A" / "B" / "C" / "C+3"
    distance: int                    # metres
    condition: str                   # e.g. "GOOD", "YIELDING"
    gate: int
    draw_weight_lb: float
    finish_time: float               # seconds
    finish_position: int
    section_times: list[float]       # per 200m
    position_calls: list[int]        # lane positions per section
    horse_weight_kg: Optional[int] = None
    jockey_id: Optional[str] = None
    interference_loss_sec: float = 0.0
    equipment_change: dict = field(default_factory=dict)


@dataclass
class RaceMetrics:
    """Computed physical metrics for one race entry."""
    race_id: str
    date: str
    venue: str
    rail: str
    distance: int
    condition: str
    gate: int
    draw_weight_lb: float
    finish_time: float
    finish_position: int
    corrected_time: float            # finish_time - interference_loss
    asr: float
    true_speed_ms: float
    fap: float
    edi: float
    fi: float
    pa: float
    section_times: list = None      # raw 200m splits (for display/verification)
    position_calls: list = None     # raw lane positions (for display/verification)

    def __post_init__(self):
        if self.section_times is None:
            self.section_times = []
        if self.position_calls is None:
            self.position_calls = []


@dataclass
class MetricSummary:
    values: list[float]
    mean: float
    std: float
    trend: str   # "improving" / "declining" / "stable"

    @classmethod
    def from_values(cls, values: list[float]) -> "MetricSummary":
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)

        if n >= 2:
            delta = values[-1] - values[0]
            if delta > std * 0.5:
                trend = "improving"
            elif delta < -std * 0.5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return cls(values=values, mean=round(mean, 4), std=round(std, 4), trend=trend)


@dataclass
class HorseProfile:
    """Rolling 3-race physical profile for a horse."""
    horse_id: str
    horse_name: str
    recent_races: list[RaceMetrics]   # ordered oldest → newest, max 3

    # Averaged metrics
    asr: Optional[MetricSummary] = None
    true_speed: Optional[MetricSummary] = None
    fap: Optional[MetricSummary] = None
    edi: Optional[float] = None
    fi: Optional[MetricSummary] = None
    pa: Optional[float] = None

    # Contextual
    preferred_distance_range: Optional[tuple[int, int]] = None
    racing_style: Optional[str] = None   # "front-runner" / "stalker" / "closer"

    def recompute_averages(self) -> None:
        """Recompute all summary stats from recent_races."""
        if not self.recent_races:
            return

        races = self.recent_races

        self.asr = MetricSummary.from_values([r.asr for r in races])
        self.true_speed = MetricSummary.from_values([r.true_speed_ms for r in races])
        self.fap = MetricSummary.from_values([r.fap for r in races])
        self.edi = round(sum(r.edi for r in races) / len(races), 2)
        self.fi = MetricSummary.from_values([r.fi for r in races])
        self.pa = round(sum(r.pa for r in races) / len(races), 4)


def compute_metrics(entry: RaceEntry) -> RaceMetrics:
    """
    Compute all physical metrics for a single race entry.

    The corrected_time removes estimated interference losses so metrics
    reflect the horse's true physical ability.
    """
    corrected_time = entry.finish_time - entry.interference_loss_sec

    asr = calculate_asr(
        distance=entry.distance,
        finish_time=corrected_time,
        actual_weight_lb=entry.draw_weight_lb,
        condition=entry.condition,
    )

    ts = calculate_true_speed(
        distance=entry.distance,
        finish_time=corrected_time,
        position_calls=entry.position_calls,
        venue=entry.venue,
    )

    fap = edi = fi = pa = 0.0
    if entry.section_times and len(entry.section_times) >= 2:
        try:
            fap = calculate_fap(entry.section_times)
        except Exception:
            pass
        try:
            edi = calculate_edi(entry.section_times)
        except Exception:
            pass
        try:
            fi = calculate_fi(entry.section_times)
        except Exception:
            pass
        try:
            pa = calculate_pa(entry.section_times, entry.distance)
        except Exception:
            pass

    return RaceMetrics(
        race_id=entry.race_id,
        date=entry.date,
        venue=entry.venue,
        rail=entry.rail,
        distance=entry.distance,
        condition=entry.condition,
        gate=entry.gate,
        draw_weight_lb=entry.draw_weight_lb,
        finish_time=entry.finish_time,
        finish_position=entry.finish_position,
        corrected_time=corrected_time,
        asr=asr,
        true_speed_ms=ts["true_speed_ms"],
        fap=fap,
        edi=edi,
        fi=fi,
        pa=pa,
        section_times=entry.section_times,
        position_calls=entry.position_calls,
    )


def update_profile(profile: HorseProfile, new_entry: RaceEntry) -> HorseProfile:
    """
    Add a new race result to a horse's profile and recompute rolling averages.
    Keeps only the most recent MAX_ROLLING_RACES races.
    """
    metrics = compute_metrics(new_entry)

    # Append and trim to rolling window
    profile.recent_races.append(metrics)
    if len(profile.recent_races) > MAX_ROLLING_RACES:
        profile.recent_races = profile.recent_races[-MAX_ROLLING_RACES:]

    profile.recompute_averages()
    return profile


def build_profile_from_entries(
    horse_id: str,
    horse_name: str,
    entries: list[RaceEntry],
) -> HorseProfile:
    """
    Build a fresh profile from a list of race entries (sorted oldest → newest).
    Only the last MAX_ROLLING_RACES entries are used.
    """
    profile = HorseProfile(
        horse_id=horse_id,
        horse_name=horse_name,
        recent_races=[],
    )
    for entry in entries[-MAX_ROLLING_RACES:]:
        metrics = compute_metrics(entry)
        profile.recent_races.append(metrics)

    profile.recompute_averages()
    return profile
