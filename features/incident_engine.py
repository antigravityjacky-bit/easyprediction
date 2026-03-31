"""
NLP Incident Engine — HKJC Chinese Report Parser
Detects "Anti-Trend" signals from race incident text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class IncidentReport:
    horse_code: str
    text: str
    signal_a_score: float = 0.0
    signal_c_score: float = 0.0
    signal_a_triggered: bool = False
    signal_c_triggered: bool = False
    incidents: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# --- Signal C: Incident Types & Impacts ---
INCIDENT_PATTERNS = [
    ("SLOW_START",  r"出閘(笨拙|僅屬一般|起步慢|起步緩慢|甩門|失地)", 0.40),
    ("BUMPED",      r"(碰撞|互相觸碰|大力觸碰|互碰)", 0.50),
    ("SQUEEZED",    r"(受擠迫|被夾|兩馬之間)", 0.50),
    ("CHECKED",     r"(收慢|受阻|避開|收韁|勒避)", 0.55),
    ("BLOCKED",     r"(受困|被困|未能望空|難以望空|無位|冇位)", 0.55),
    ("STUMBLED",    r"(失去平衡|失蹄)", 0.60),
    ("LOST_SHOE",   r"(失去蹄鐵|甩蹄鐵)", 0.35),
    ("INTERFERENCE", r"(受干擾|被帶|斜跑致受)", 0.40),
    ("LOST_GROUND", r"(失去位置|墮後|蝕位)", 0.40),
    ("HUNG",        r"(外閃|內閃)", 0.25),
    ("INJURY",      r"(割傷|受傷)", 0.35), # Often found post-race
    ("KEEN",        r"搶口", 0.20),
]

# --- Signal A: Wide No Cover Patterns ---
WIDE_PATTERNS = r"([三四五六])疊|外疊|大外|外側|外檔競跑|繞過|兜外彎"
NO_COVER_PATTERNS = r"無遮擋|沒有遮擋|冇遮擋|獨自在外|無馬帶"

# --- Stage Markers ---
STAGE_MARKERS = {
    "early": r"起步時|開始時|首\d+米|早段",
    "mid":   r"對面直路|中段|轉彎時|彎位|過路口",
    "late":  r"直路上|最後\d+米|末段|終點前",
}


def parse_horse_incidents(horse_code: str, report_text: str, running_style: str = "midfield", 
                         finishing_pos: int = 1, field_size: int = 12, lbw: float = 0.0) -> IncidentReport:
    """
    Parse a Chinese stewards' report for a specific horse and calculate A/C signals.
    """
    report = IncidentReport(horse_code=horse_code, text=report_text)
    if not report_text or len(report_text) < 5:
        return report

    # 1. Detect Signal A: Wide No Cover
    is_wide = re.search(WIDE_PATTERNS, report_text)
    no_cover = re.search(NO_COVER_PATTERNS, report_text)
    
    if is_wide:
        # Extract number of layers if available
        layers = 3
        layer_match = re.search(r"([三四五六])疊", report_text)
        if layer_match:
            mapping = {"三": 3, "四": 4, "五": 5, "六": 6}
            layers = mapping.get(layer_match.group(1), 3)
        
        # Scoring Logic for Signal A
        layer_weight = {3: 1.0, 4: 1.5, 5: 2.0, 6: 2.5}.get(layers, 1.0)
        no_cover_bonus = 1.5 if no_cover else 1.0
        
        # Style Multiplier
        style_impact_multiplier = {
            "leader":    1.8,
            "prominent": 1.5,
            "midfield":  1.0,
            "closer":    0.3,
        }.get(running_style, 1.0)
        
        # Performance Factor
        finish_percentile = 1 - (finishing_pos / field_size)
        margin_bonus = max(0, 1 - lbw * 0.2)
        performance = finish_percentile * 0.6 + margin_bonus * 0.4
        
        # confidence (set to 1.0 for now unless text is ambiguous)
        confidence = 1.0
        
        score_a = layer_weight * no_cover_bonus * style_impact_multiplier * performance * confidence
        report.signal_a_score = round(min(score_a * 2.5, 10.0), 2)
        if report.signal_a_score >= 4.0: # Threshold for trigger
            report.signal_a_triggered = True

    # 2. Detect Signal C: Multiple Incidents
    found_types = set()
    found_stages = set()
    total_impact = 0.0
    
    for inc_type, pattern, impact in INCIDENT_PATTERNS:
        match = re.search(pattern, report_text)
        if match:
            # Special adjustment for certain phrases
            current_impact = impact
            if "僅屬一般" in match.group(0):
                current_impact = 0.20
            
            # Determine stage
            stage = "mid" # default
            for s_name, s_pat in STAGE_MARKERS.items():
                # Look for stage marker within 30 chars of the incident
                start = max(0, match.start() - 30)
                end = min(len(report_text), match.end() + 30)
                context = report_text[start:end]
                if re.search(s_pat, context):
                    stage = s_name
                    break
            
            found_types.add(inc_type)
            found_stages.add(stage)
            total_impact += current_impact
            report.incidents.append({"type": inc_type, "impact": current_impact, "stage": stage})

    if len(found_types) >= 2 and finishing_pos <= 3:
        diversity_mult = 1.0 + (len(found_types) - 1) * 0.2
        stage_mult = 1.0 + (len(found_stages) - 1) * 0.15
        pos_score = {1: 2.5, 2: 1.8, 3: 1.2}.get(finishing_pos, 1.0)
        margin_bonus = max(0, 1 - lbw * 0.15)
        
        score_c = total_impact * diversity_mult * stage_mult * (pos_score + margin_bonus)
        report.signal_c_score = round(min(score_c * 1.5, 10.0), 2)
        report.signal_c_triggered = True

    return report
