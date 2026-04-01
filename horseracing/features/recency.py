"""
Recency Features — 編碼衰減歷史訊號

馬匹「最近狀態」比「長期平均」更重要。
本模塊提供加權的最近 N 場成績。

特徵列表:
  - recent_3races_win_rate: 最近 3 場的勝率（衰減加權）
  - recent_6races_trend: 最近 6 場的趨勢方向
  - form_momentum: 最近成績的加速度（二階導數）
  - layoff_penalty: days_since_last 的非線性懲罰
  - recency_strength: 最近狀態有多強（0-1）
"""

from __future__ import annotations

import numpy as np


def calc_recent_3races_win_rate(
    finish_positions: list[int] | None,
    decay_factors: tuple[float, float, float] = (0.7, 0.5, 1.0),
) -> float:
    """
    計算最近 3 場的加權勝率。

    參數:
      finish_positions: 最近 3 場的完成位置 [pos1, pos2, pos3]
                        其中 pos1 是最新的一場
      decay_factors: 衰減因子 (最新的加權最高)
                    默認: 最新場次權重最高（1.0）

    返回:
      float [0-1] 最近 3 場的加權勝率
    """
    if not finish_positions or len(finish_positions) == 0:
        return 0.5  # 無歷史

    # 補齊到 3 場（不足的用 None）
    positions = list(finish_positions[-3:])
    while len(positions) < 3:
        positions.insert(0, None)

    # 計算每場的勝敗 (< 4 為勝)
    results = []
    for i, pos in enumerate(positions):
        if pos is None:
            continue
        is_top3 = 1 if pos <= 3 else 0
        weight = decay_factors[i]
        results.append((is_top3, weight))

    if not results:
        return 0.5

    total_weight = sum(w for _, w in results)
    total_wins = sum(result * weight for result, weight in results)

    win_rate = total_wins / total_weight
    return min(1.0, max(0.0, win_rate))


def calc_recent_6races_trend(
    finish_positions: list[int] | None,
) -> float:
    """
    計算最近 6 場的趨勢方向。

    返回:
      float [-1, 1]
        +1: 上升趨勢（最近表現更好）
        0: 平穩
        -1: 下降趨勢（最近表現更差）
    """
    if not finish_positions or len(finish_positions) < 2:
        return 0.0

    positions = list(finish_positions[-6:])
    if len(positions) < 2:
        return 0.0

    # 計算成績序列（top3 = 1, other = 0）
    results = [1 if p <= 3 else 0 for p in positions]

    # 簡單線性趨勢：最近 3 場 vs 之前 3 場
    if len(results) >= 6:
        recent_3 = sum(results[-3:]) / 3.0
        earlier_3 = sum(results[:3]) / 3.0
        trend = recent_3 - earlier_3
    else:
        # 少於 6 場，計算整體趨勢
        avg = sum(results) / len(results)
        trend = results[-1] - avg

    return min(1.0, max(-1.0, trend * 2))  # 縮放到 [-1, 1]


def calc_form_momentum(
    finish_positions: list[int] | None,
) -> float:
    """
    計算最近成績的加速度（momentum）。

    返回:
      float [-1, 1]
        +1: 加速上升（連勝或不斷進步）
        0: 無加速
        -1: 加速下降（連敗或不斷退步）
    """
    if not finish_positions or len(finish_positions) < 3:
        return 0.0

    positions = list(finish_positions[-4:])
    if len(positions) < 3:
        return 0.0

    # 計算成績序列
    results = [1 if p <= 3 else 0 for p in positions]

    # 二階差分：加速度
    if len(results) == 4:
        # 0-1 差, 1-2 差, 2-3 差
        diff1 = results[1] - results[0]
        diff2 = results[2] - results[1]
        diff3 = results[3] - results[2]

        # 加速度 = 二階差分
        accel = (diff3 - diff1) / 2.0  # [-1, 1]
        return accel
    else:
        # 少於 4 場，使用簡單差分
        diff = results[-1] - results[0]
        return min(1.0, max(-1.0, diff))


def calc_layoff_penalty(days_since_last: int | None) -> float:
    """
    計算因長時間不跑而受到的懲罰。

    邏輯:
      - 0-7 天: 無懲罰 (1.0)
      - 8-14 天: 輕微獎勵 (1.05) — 狀態調整
      - 15-30 天: 輕微懲罰 (0.95)
      - 30-60 天: 中等懲罰 (0.85)
      - 60+ 天: 重懲罰 (0.6)

    返回:
      float [0-1] 乘以該系數來調整特徵
    """
    if days_since_last is None or days_since_last <= 0:
        return 1.0

    if days_since_last <= 7:
        return 1.0
    elif days_since_last <= 14:
        return 1.05  # 輕微獎勵
    elif days_since_last <= 30:
        return 0.95
    elif days_since_last <= 60:
        return 0.85
    else:
        return 0.6


def calc_recency_strength(
    recent_3races_win_rate: float,
    form_momentum: float,
    layoff_penalty_factor: float,
) -> float:
    """
    計算「最近狀態有多強」的綜合指標。

    返回:
      float [0-1] 最近狀態強度
    """
    # 結合最近勝率 + 動量 + 休息懲罰
    win_component = recent_3races_win_rate * 0.6
    momentum_component = (form_momentum + 1.0) / 2.0 * 0.2  # 轉為 [0-1]
    layoff_component = layoff_penalty_factor * 0.2

    strength = win_component + momentum_component + layoff_component
    return min(1.0, max(0.0, strength))


def extract_recency_features(
    finish_positions: list[int] | None,
    days_since_last: int | None,
) -> dict[str, float]:
    """
    為單匹馬提取所有衰減歷史特徵。

    返回:
      {
        'recent_3races_win_rate': float,
        'recent_6races_trend': float,
        'form_momentum': float,
        'layoff_penalty': float,
        'recency_strength': float,
      }
    """
    recent_3_wr = calc_recent_3races_win_rate(finish_positions)
    recent_6_trend = calc_recent_6races_trend(finish_positions)
    momentum = calc_form_momentum(finish_positions)
    layoff_penalty = calc_layoff_penalty(days_since_last)
    recency_strength = calc_recency_strength(recent_3_wr, momentum, layoff_penalty)

    return {
        'recent_3races_win_rate': recent_3_wr,
        'recent_6races_trend': recent_6_trend,
        'form_momentum': momentum,
        'layoff_penalty': layoff_penalty,
        'recency_strength': recency_strength,
    }
