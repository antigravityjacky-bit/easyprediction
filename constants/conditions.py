"""
Track condition correction factors and gate impact coefficients.
"""

from __future__ import annotations


# Keys match HKJC official condition strings
TRACK_CONDITION_FACTORS = {
    "GOOD": {
        "speed_factor": 1.000,
        "friction": 0.42,
        "weight_impact": 1.00,
        "stamina_cost": 1.00,
    },
    "GOOD_TO_FIRM": {
        "speed_factor": 0.980,   # 2% faster than GOOD
        "friction": 0.38,
        "weight_impact": 0.95,   # lighter horses benefit
        "stamina_cost": 0.98,
    },
    "GOOD_TO_FIRM_GOOD": {      # intermediate
        "speed_factor": 0.990,
        "friction": 0.40,
        "weight_impact": 0.97,
        "stamina_cost": 0.99,
    },
    "GOOD_TO_YIELDING": {
        "speed_factor": 1.015,
        "friction": 0.47,
        "weight_impact": 1.04,
        "stamina_cost": 1.08,
    },
    "YIELDING": {
        "speed_factor": 1.030,
        "friction": 0.51,
        "weight_impact": 1.08,   # large-hoofed / heavier horses benefit
        "stamina_cost": 1.15,
    },
    "SLOW": {
        "speed_factor": 1.060,
        "friction": 0.58,
        "weight_impact": 1.12,
        "stamina_cost": 1.25,
    },
}

# Aliases mapping common display strings to canonical keys
CONDITION_ALIASES = {
    "好地": "GOOD",
    "快地": "GOOD_TO_FIRM",
    "好地至快地": "GOOD_TO_FIRM_GOOD",
    "好地至黏地": "GOOD_TO_YIELDING",
    "黏地": "YIELDING",
    "濕慢": "SLOW",
    "good": "GOOD",
    "firm": "GOOD_TO_FIRM",
    "good to firm": "GOOD_TO_FIRM",
    "good to yielding": "GOOD_TO_YIELDING",
    "yielding": "YIELDING",
    "slow": "SLOW",
}

# Gate impact coefficients: seconds lost per lane position above 1
# (gate - 1) * coefficient = additional seconds
GATE_IMPACT_COEFFICIENTS = {
    "SHA_TIN": {
        "A":   0.08,
        "A+3": 0.095,   # interpolated: 75% between A(30.5m) and B(26.5m) by width
        "B":   0.10,
        "B+2": 0.115,   # interpolated: 50% between B(26.5m) and C(22.5m) by width
        "C":   0.13,
        "C+3": 0.18,
    },
    "HAPPY_VALLEY": {
        "A":   0.15,
        "A+3": 0.17,    # interpolated: 75% between A and B
        "B":   0.18,
        "B+2": 0.22,    # interpolated: 50% between B and C
        "C":   0.25,
        "C+3": 0.35,
    },
}


def get_condition_key(condition_str: str) -> str:
    """Normalise a condition string to a canonical key."""
    key = condition_str.strip()
    if key in TRACK_CONDITION_FACTORS:
        return key
    alias = CONDITION_ALIASES.get(key.lower(), CONDITION_ALIASES.get(key))
    if alias:
        return alias
    raise ValueError(f"Unknown track condition: '{condition_str}'. "
                     f"Known aliases: {list(CONDITION_ALIASES.keys())}")


def get_speed_factor(condition_str: str) -> float:
    """Return speed_factor for a given condition string."""
    return TRACK_CONDITION_FACTORS[get_condition_key(condition_str)]["speed_factor"]
