"""
Stewards' Report Parser
Extracts interference events and estimates time loss in seconds.
"""

from __future__ import annotations


import re
from horseracing.constants.equipment import INTERFERENCE_LOSS


# Keywords → interference type mapping
KEYWORD_MAP = [
    (["bump", "bumped", "collision", "碰撞"],  "gate_bump"),
    (["checked", "collect", "收慢"],            "checked"),
    (["trapped", "boxed", "困"],               "trapped_inside"),
    (["pulled", "pull hard", "搶口"],           "pulling"),
    (["lost whip", "掉鞭"],                     "lost_whip"),
    (["pushed wide", "out", "帶外", "帶出"],    "pushed_wide"),
]

SEVERITY_MAP = {
    "gate_bump":      {"minor": 0.10, "moderate": 0.20, "severe": 0.40},
    "checked":        {"once": 0.25, "multiple": 0.50},
    "trapped_inside": {"brief": 0.30, "long": 0.80},
    "pulling":        {"minor": 0.20, "moderate": 0.50, "severe": 1.00},
    "lost_whip":      {"": 0.35},
    "pushed_wide":    {"per_lane": 0.15},
}

SEVERITY_KEYWORDS = {
    "severe":   ["severely", "seriously", "badly", "嚴重"],
    "moderate": ["moderate", "badly", "中度"],
    "minor":    ["slightly", "briefly", "輕微", "minor"],
    "multiple": ["repeatedly", "again", "多次"],
}


def _detect_severity(text: str) -> str:
    text_lower = text.lower()
    for sev, keywords in SEVERITY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return sev
    return "moderate"


def _count_lanes(text: str) -> int:
    match = re.search(r"(\d+)\s+lane", text, re.IGNORECASE)
    return int(match.group(1)) if match else 1


def parse_interference(report_text: str) -> dict:
    """
    Parse a free-text stewards' report for a single horse.

    Parameters
    ----------
    report_text : str
        Text from HKJC stewards' inquiry / note column.

    Returns
    -------
    dict with keys:
        incidents: list of {type, severity, estimated_loss_sec}
        total_loss_sec: float
    """
    incidents = []
    total_loss = 0.0
    text_lower = report_text.lower()

    for keywords, incident_type in KEYWORD_MAP:
        if not any(k.lower() in text_lower for k in keywords):
            continue

        severity = _detect_severity(report_text)
        sev_table = SEVERITY_MAP.get(incident_type, {})

        if incident_type == "pushed_wide":
            lanes = _count_lanes(report_text)
            loss = sev_table["per_lane"] * lanes
        elif severity in sev_table:
            loss = sev_table[severity]
        elif sev_table:
            # fallback to first value
            loss = list(sev_table.values())[0]
        else:
            loss = 0.0

        incidents.append({
            "type": incident_type,
            "severity": severity,
            "estimated_loss_sec": loss,
        })
        total_loss += loss

    return {
        "incidents": incidents,
        "total_loss_sec": round(total_loss, 2),
    }


def estimate_total_loss(report_text: str) -> float:
    """Convenience function returning only the total estimated time loss (seconds)."""
    return parse_interference(report_text)["total_loss_sec"]
