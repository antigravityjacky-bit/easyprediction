"""
Track physical constants for Sha Tin and Happy Valley racecourses.
All measurements in metres and seconds.
"""

from __future__ import annotations


SHA_TIN_TURF = {
    "circumference": 1900,       # metres (inner rail baseline)
    "straight_length": 430,      # metres (constant across all rail positions)
    "direction": "right",
    "rails": {
        "A": {
            "width": 30.5,
            "circumference": 1900,
            "bend_radius": 80,
            "note": "Widest — most equitable draw",
        },
        "A+3": {
            "width": 27.5,       # A rail narrowed by 3m (user confirmed)
            "circumference": 1896,  # interpolated A→B
            "bend_radius": 79,
            "note": "A rail moved in 3m — slight inside bias",
        },
        "B": {
            "width": 26.5,
            "circumference": 1895,
            "bend_radius": 78,
            "note": "Medium width",
        },
        "B+2": {
            "width": 24.5,       # B rail narrowed by 2m (user confirmed)
            "circumference": 1893,  # interpolated B→C
            "bend_radius": 78,
            "note": "B rail moved in 2m — moderate inside bias",
        },
        "C": {
            "width": 22.5,
            "circumference": 1890,
            "bend_radius": 77,
            "note": "Narrow — inside draw advantage emerges",
        },
        "C+3": {
            "width": 18.3,
            "circumference": 1885,
            "bend_radius": 76,
            "note": "Narrowest — extreme front-runner advantage",
        },
    },
    # bend_ratio = bend_length / total_distance
    "distances": {
        1000: {"num_bends": 0, "bend_length": 0,   "bend_ratio": 0.000},
        1200: {"num_bends": 1, "bend_length": 485,  "bend_ratio": 0.404},
        1400: {"num_bends": 1, "bend_length": 485,  "bend_ratio": 0.346},
        1600: {"num_bends": 1, "bend_length": 485,  "bend_ratio": 0.303},
        1650: {"num_bends": 1, "bend_length": 485,  "bend_ratio": 0.294},
        1800: {"num_bends": 2, "bend_length": 970,  "bend_ratio": 0.539},
        2000: {"num_bends": 2, "bend_length": 970,  "bend_ratio": 0.485},
        2200: {"num_bends": 2, "bend_length": 970,  "bend_ratio": 0.441},
        2400: {"num_bends": 2, "bend_length": 970,  "bend_ratio": 0.404},
    },
}

HAPPY_VALLEY_TURF = {
    "circumference": 1450,       # metres (baseline)
    "direction": "right",
    "rails": {
        "A": {
            "width": 30.5,
            "straight_length": 312,
            "bend_radius": 52,
            "note": "Widest — closers have room",
        },
        "A+3": {
            "width": 27.5,       # A rail narrowed by 3m (user confirmed)
            "straight_length": 314,  # interpolated A→B
            "bend_radius": 53,
            "note": "A rail moved in 3m",
        },
        "B": {
            "width": 26.5,
            "straight_length": 315,
            "bend_radius": 54,
            "circumference": 1435,
            "note": "Medium — slight pace-setter bias",
        },
        "B+2": {
            "width": 24.5,       # B rail narrowed by 2m (user confirmed)
            "straight_length": 323,  # interpolated B→C
            "bend_radius": 51,
            "note": "B rail moved in 2m — moderate inside bias",
        },
        "C": {
            "width": 22.5,
            "straight_length": 330,
            "bend_radius": 48,
            "note": "Narrow — large inside draw advantage",
        },
        "C+3": {
            "width": 19.5,
            "straight_length": 335,
            "bend_radius": 45,
            "note": "Narrowest — front-runner benefit extreme",
        },
    },
    "distances": {
        1000: {"num_bends": 1, "bend_length": 402,  "bend_ratio": 0.402},
        1200: {"num_bends": 2, "bend_length": 804,  "bend_ratio": 0.670},
        1650: {"num_bends": 2, "bend_length": 900,  "bend_ratio": 0.545},
        1800: {"num_bends": 2, "bend_length": 964,  "bend_ratio": 0.536},
        2200: {"num_bends": 3, "bend_length": 1148, "bend_ratio": 0.522},
    },
    "gradient": {
        "straight_incline_deg": 0.3,
        "height_diff_m": 1.65,
        "time_penalty_sec": 0.95,
    },
}

# Straight length per venue/rail (used by sprint_capability)
STRAIGHT_LENGTHS = {
    "SHA_TIN": {
        "A": 430, "A+3": 430, "B": 430, "B+2": 430, "C": 430, "C+3": 430,
    },
    "HAPPY_VALLEY": {
        "A": 312, "A+3": 314, "B": 315, "B+2": 323, "C": 330, "C+3": 335,
    },
}

TRACK_DATA = {
    "SHA_TIN": SHA_TIN_TURF,
    "HAPPY_VALLEY": HAPPY_VALLEY_TURF,
}


def get_bend_ratio(venue: str, distance: int) -> float:
    """Return the bend ratio for a given venue and distance."""
    data = TRACK_DATA[venue]
    distances = data["distances"]
    if distance not in distances:
        raise ValueError(f"Distance {distance}m not defined for {venue}. "
                         f"Available: {sorted(distances.keys())}")
    return distances[distance]["bend_ratio"]


def get_straight_length(venue: str, rail: str) -> int:
    """Return straight length in metres for a given venue and rail."""
    return STRAIGHT_LENGTHS[venue][rail]


def get_rail_width(venue: str, rail: str) -> float:
    """Return track width in metres for a given venue and rail."""
    return TRACK_DATA[venue]["rails"][rail]["width"]
