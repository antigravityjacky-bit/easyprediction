"""
Pairing Features — 編碼騎師-馬匹配對訊號

特定騎師與特定馬匹的組合可能有獨特的優勢。
本模塊捕捉這一訊號。

特徵列表:
  - jockey_horse_affinity: 騎師與該馬的歷史成績比率
  - jockey_recent_form: 騎師最近的狀態
  - jockey_distance_affinity: 騎師在該距離的特殊能力
  - pairing_sample_size: 配對的歷史樣本數（置信度調整）
"""

from __future__ import annotations

from collections import defaultdict


def calc_jockey_horse_affinity(
    jockey_name: str,
    horse_code: str,
    jockey_horse_history: dict[tuple[str, str], list[int]] | None,
) -> float:
    """
    計算騎師與馬匹的「親和力」。

    參數:
      jockey_name: 騎師名字
      horse_code: 馬匹代碼
      jockey_horse_history: {(jockey, horse): [positions]} 配對歷史

    返回:
      float [0-1] 該配對的命中率（top-3）
    """
    if not jockey_horse_history:
        return 0.5

    key = (jockey_name, horse_code)
    positions = jockey_horse_history.get(key, [])

    if not positions:
        return 0.5  # 無配對歷史

    # 計算命中率 (top-3)
    hits = sum(1 for p in positions if p <= 3)
    affinity = hits / len(positions)

    # 應用置信度調整：樣本少時更接近 0.5
    sample_count = len(positions)
    confidence_weight = min(1.0, sample_count / 5.0)  # 5+ 樣本時達到最高置信度

    adjusted_affinity = 0.5 + (affinity - 0.5) * confidence_weight

    return adjusted_affinity


def calc_jockey_recent_form(
    jockey_name: str,
    jockey_recent_results: dict[str, list[int]] | None,
    window_races: int = 10,
) -> float:
    """
    計算騎師的最近狀態。

    參數:
      jockey_name: 騎師名字
      jockey_recent_results: {jockey: [positions]} 最近賽事成績
      window_races: 回看窗口

    返回:
      float [0-1] 騎師的最近成績勝率
    """
    if not jockey_recent_results:
        return 0.5

    positions = jockey_recent_results.get(jockey_name, [])
    if not positions:
        return 0.5

    # 限制窗口
    recent = positions[-window_races:]

    # 計算最近成績的 top-3 率
    hits = sum(1 for p in recent if p <= 3)
    win_rate = hits / len(recent)

    return win_rate


def calc_jockey_distance_affinity(
    jockey_name: str,
    distance: int,
    jockey_distance_stats: dict[tuple[str, int], dict] | None,
) -> float:
    """
    計算騎師在特定距離上的特殊能力。

    參數:
      jockey_name: 騎師名字
      distance: 賽事距離 (m)
      jockey_distance_stats: {(jockey, distance): {win_rate, count}} 距離特化成績

    返回:
      float [-0.2, 0.2] 相對於騎師平均的優勢/劣勢
    """
    if not jockey_distance_stats:
        return 0.0

    key = (jockey_name, distance)
    stats = jockey_distance_stats.get(key)

    if not stats or stats.get('count', 0) < 3:
        return 0.0  # 樣本不足

    distance_win_rate = stats.get('win_rate', 0.5)
    overall_win_rate = stats.get('overall_win_rate', 0.5)

    # 返回差異（縮放到 [-0.2, 0.2]）
    diff = distance_win_rate - overall_win_rate
    return min(0.2, max(-0.2, diff))


def calc_pairing_confidence(
    jockey_horse_sample_count: int,
    jockey_sample_count: int,
) -> float:
    """
    計算騎師-馬配對特徵的置信度。

    邏輯:
      - 配對樣本 >= 5: 置信度高 (0.9)
      - 配對樣本 >= 3: 置信度中 (0.7)
      - 配對樣本 >= 1: 置信度低 (0.4)
      - 無配對樣本: 置信度極低 (0.1)

    返回:
      float [0-1] 置信度乘數
    """
    if jockey_horse_sample_count >= 5:
        return 1.0
    elif jockey_horse_sample_count >= 3:
        return 0.8
    elif jockey_horse_sample_count >= 1:
        return 0.5
    else:
        return 0.2


def extract_pairing_features(
    jockey_name: str,
    horse_code: str,
    distance: int,
    jockey_horse_history: dict[tuple[str, str], list[int]] | None = None,
    jockey_recent_results: dict[str, list[int]] | None = None,
    jockey_distance_stats: dict[tuple[str, int], dict] | None = None,
) -> dict[str, float]:
    """
    為單匹馬提取所有騎師配對特徵。

    返回:
      {
        'jockey_horse_affinity': float,
        'jockey_recent_form': float,
        'jockey_distance_affinity': float,
        'pairing_confidence': float,
      }
    """
    affinity = calc_jockey_horse_affinity(jockey_name, horse_code, jockey_horse_history)
    recent_form = calc_jockey_recent_form(jockey_name, jockey_recent_results)
    dist_affinity = calc_jockey_distance_affinity(jockey_name, distance, jockey_distance_stats)

    # 計算配對樣本數
    key = (jockey_name, horse_code)
    jockey_horse_samples = len(jockey_horse_history.get(key, []) if jockey_horse_history else [])
    jockey_samples = len(jockey_recent_results.get(jockey_name, []) if jockey_recent_results else [])

    confidence = calc_pairing_confidence(jockey_horse_samples, jockey_samples)

    return {
        'jockey_horse_affinity': affinity,
        'jockey_recent_form': recent_form,
        'jockey_distance_affinity': dist_affinity,
        'pairing_confidence': confidence,
    }
