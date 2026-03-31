"""
Physical Performance Metrics — Running Style & Pace Analysis
"""

from __future__ import annotations

from horseracing.constants.standard_times import get_standard_time, get_z_score


def classify_running_style(position_calls: list[int], field_size: int) -> str:
    """
    Classify horse running style based on positional percentile.
    """
    if not position_calls or field_size <= 0:
        return "midfield"
    
    avg_pos = sum(position_calls) / len(position_calls)
    percentile = avg_pos / field_size
    
    if percentile <= 0.20:
        return "leader"
    elif percentile <= 0.40:
        return "prominent"
    elif percentile <= 0.70:
        return "midfield"
    else:
        return "closer"


def calculate_signal_b(finish_time: float, venue: str, track: str, distance: int, 
                       cls: str, finish_pos: int, field_size: int) -> dict:
    """
    Calculate Signal B: Fast Pace Top 3 performance.
    """
    z_score = get_z_score(finish_time, venue, track, distance, cls)
    
    result = {
        "z_score": z_score,
        "triggered": False,
        "score": 0.0
    }
    
    if z_score is None:
        return result
    
    # Threshold check: Z <= -1.0 (Fast) AND finished in top 3
    if z_score <= -1.0 and finish_pos <= 3:
        # Severity tiers
        if z_score <= -2.0:
            severity_mult = 2.0 # Extreme
        elif z_score <= -1.5:
            severity_mult = 1.5 # Very Fast
        else:
            severity_mult = 1.0 # Fast
        
        # Style multiplier
        # Closers are harder to win in fast paces, Leaders are harder to maintain
        # Prompt says: "重點在於該馬匹在快賽事中展現了超越其名次的實力"
        # We assign a base score for Signal B
        base_score = 4.0
        
        # Position bonus
        pos_bonus = {1: 1.5, 2: 1.2, 3: 1.0}.get(finish_pos, 1.0)
        
        final_score = base_score * severity_mult * pos_bonus
        result["score"] = round(min(final_score, 10.0), 2)
        result["triggered"] = True
        
    return result
