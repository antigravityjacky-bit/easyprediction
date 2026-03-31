"""
Equipment effects and jockey performance adjustments (in seconds).
Negative = faster, Positive = slower.
"""

from __future__ import annotations


EQUIPMENT_EFFECTS = {
    "blinkers_half_first_time": {
        "description": "Half blinkers (H), first time — horse that tends to hang/pull",
        "pullers": -0.50,
        "distracted": -0.30,
        "normal": 0.00,
    },
    "blinkers_full_first_time": {
        "description": "Full blinkers, first time",
        "effect": -0.80,
        "risk_note": "May cause slow reaction in gate",
    },
    "crossover_noseband": {
        "description": "Crossover noseband (XB) — improves breathing",
        "effect": -0.20,
    },
    "noseband": {
        "description": "Standard noseband",
        "effect": -0.10,
    },
    "tongue_tie": {
        "description": "Tongue tie (TT) — prevents swallowing tongue",
        "effect": -0.15,
    },
    "blinkers_removed_first_time": {
        "description": "Blinkers removed for first time",
        "effect": +0.30,   # may distract
    },
    "noseband_removed_first_time": {
        "description": "Noseband removed for first time",
        "effect": +0.10,
    },
    # Combinations (non-additive — synergy captured here)
    "blinkers_half_plus_crossover": {
        "description": "H + XB combined",
        "effect": -0.70,
    },
    "noseband_plus_tongue_tie": {
        "description": "Noseband + TT combined",
        "effect": -0.40,
    },
}

# Jockey time advantage relative to average jockey level (seconds).
# Adjustments only applied when jockey changes between reference race and target race.
JOCKEY_EFFECT = {
    "潘頓":   -0.15,
    "莫雷拉":  -0.12,
    "巴度":   -0.08,
    "艾兆禮":  -0.05,
    "布文":   -0.05,
    # Apprentice handling: net of -2lb weight allowance minus skill deficit
    "_apprentice_net": +0.116,  # +0.20 skill loss - 0.084 weight gain
}

# Interference loss lookup (seconds) used to correct historical race times
INTERFERENCE_LOSS = {
    "gate_bump_minor":      0.10,
    "gate_bump_moderate":   0.20,
    "gate_bump_severe":     0.40,
    "checked_once":         0.25,
    "checked_multiple":     0.50,
    "trapped_inside_brief": 0.30,
    "trapped_inside_long":  0.80,
    "pulling_minor":        0.20,
    "pulling_moderate":     0.50,
    "pulling_severe":       1.00,
    "lost_whip":            0.35,
    "pushed_wide_per_lane": 0.15,   # multiply by number of lanes pushed out
}


def get_equipment_effect(equipment_dict: dict) -> float:
    """
    Calculate total equipment time adjustment from a horse's equipment change dict.

    Parameters
    ----------
    equipment_dict : dict
        Keys are equipment effect names from EQUIPMENT_EFFECTS, values are True/False
        or a specific sub-type string (e.g. {"blinkers_half_first_time": "pullers"}).

    Returns
    -------
    float
        Total seconds adjustment (negative = faster).
    """
    total = 0.0
    for key, value in equipment_dict.items():
        if key not in EQUIPMENT_EFFECTS:
            continue
        entry = EQUIPMENT_EFFECTS[key]
        if isinstance(value, str) and value in entry:
            total += entry[value]
        elif "effect" in entry:
            total += entry["effect"]
    return total
