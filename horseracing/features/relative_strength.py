"""
Relative Strength Features — 編碼馬匹相對於同場的強度

「馬匹絕對能力」不如「相對於同場對手的優勢」重要。
本模塊捕捉馬匹在同場中的相對地位。

特徵列表:
  - vs_field_avg_speed: 該馬速度 vs 同場平均
  - vs_field_win_rate: 該馬歷史勝率 vs 同場平均
  - field_dominance_score: 該馬在同場中的統治力評分
  - upset_potential: 低等級但能力強的黑馬信號
  - favorite_indicator: 該馬是否為「大熱馬」或「冷馬」
"""

from __future__ import annotations

import numpy as np


def calc_vs_field_avg_speed(
    horse_speed: float,
    field_speeds: list[float],
) -> float:
    """
    計算該馬速度相對於同場平均的差異。

    參數:
      horse_speed: 該馬的平均速度
      field_speeds: 同場所有馬的速度列表

    返回:
      float [-1, 1] 相對優勢
        +1: 該馬遠快於同場平均
        0: 與同場平均相當
        -1: 該馬遠慢於同場平均
    """
    if not field_speeds or len(field_speeds) == 0:
        return 0.0

    field_avg = sum(field_speeds) / len(field_speeds)

    if field_avg == 0:
        return 0.0

    # 標準化的相對差異
    relative_diff = (horse_speed - field_avg) / field_avg
    # 縮放到 [-1, 1]
    return min(1.0, max(-1.0, relative_diff * 2))


def calc_vs_field_win_rate(
    horse_win_rate: float,
    field_win_rates: list[float],
) -> float:
    """
    計算該馬歷史勝率相對於同場平均。

    返回:
      float [-1, 1] 相對優勢
        +1: 該馬明顯優於同場
        0: 與同場相當
        -1: 該馬明顯劣於同場
    """
    if not field_win_rates or len(field_win_rates) == 0:
        return 0.0

    field_avg = sum(field_win_rates) / len(field_win_rates)

    relative_diff = horse_win_rate - field_avg
    # 縮放到 [-1, 1]，考慮到勝率是概率
    return min(1.0, max(-1.0, relative_diff * 3))


def calc_field_dominance_score(
    horse_speed_advantage: float,
    horse_win_rate_advantage: float,
    horse_recent_form: float,
) -> float:
    """
    計算該馬在同場中的「統治力」評分。

    邏輯:
      結合速度優勢、勝率優勢、最近狀態，計算馬匹的絕對統治力。

    返回:
      float [0-1] 統治力評分
        0.8-1.0: 該馬明顯統治同場
        0.5-0.8: 該馬有明顯優勢
        0.3-0.5: 該馬競爭力一般
        0-0.3: 該馬明顯劣勢
    """
    # 標準化到 [0-1]
    speed_norm = (horse_speed_advantage + 1) / 2.0  # [-1, 1] → [0, 1]
    wr_norm = (horse_win_rate_advantage + 1) / 2.0  # [-1, 1] → [0, 1]

    # 加權組合
    dominance = speed_norm * 0.4 + wr_norm * 0.3 + horse_recent_form * 0.3

    return min(1.0, max(0.0, dominance))


def calc_upset_potential(
    horse_class_rank: int | None,
    field_avg_class: float,
    horse_actual_speed: float,
    field_avg_speed: float,
) -> float:
    """
    計算馬匹的「黑馬潛力」——等級低但能力強。

    邏輯:
      - 如果馬匹等級低（rank 高）但實際速度高 → 黑馬潛力大
      - 反之 → 黑馬潛力小

    參數:
      horse_class_rank: 該馬的等級排名（越小越好）
      field_avg_class: 同場平均等級
      horse_actual_speed: 該馬的實際速度
      field_avg_speed: 同場平均速度

    返回:
      float [-1, 1]
        +1: 高黑馬潛力（被低估的馬）
        0: 合理估值
        -1: 被高估的馬（等級高但速度不足）
    """
    if horse_class_rank is None:
        return 0.0

    # 等級評分：低等級 (高排名) = 低分
    class_expectation = horse_class_rank / 14.0  # 歸一化 [0-1]

    # 實際表現評分
    if field_avg_speed > 0:
        actual_performance = horse_actual_speed / field_avg_speed
    else:
        actual_performance = 0.5

    # 潛力 = 實際表現 - 等級期望
    potential = (actual_performance - 0.5) * 2 - class_expectation + 0.5
    return min(1.0, max(-1.0, potential))


def calc_favorite_indicator(
    horse_selection_count_a: int,
    horse_selection_count_b: int,
    field_size: int,
) -> float:
    """
    計算該馬是否為「大熱馬」或「冷馬」。

    邏輯:
      - 被 A & B 都多次選中 → 大熱馬 (1.0)
      - 甚少被選中 → 冷馬 (-1.0)

    返回:
      float [-1, 1]
        +1: 大熱馬（被看好）
        0: 中等馬
        -1: 冷馬（被看淡）
    """
    # 計算被選中的頻率
    total_selections = horse_selection_count_a + horse_selection_count_b
    max_possible_selections = field_size * 2  # 每匹馬在 A & B 都可能被選

    selection_rate = total_selections / max_possible_selections if max_possible_selections > 0 else 0.5

    # 轉為 [-1, 1]
    favorite = (selection_rate - 0.5) * 2
    return min(1.0, max(-1.0, favorite))


def extract_relative_strength_features(
    horse_speed: float,
    horse_win_rate: float,
    horse_recent_form: float,
    horse_class_rank: int | None,
    horse_selection_count_a: int,
    horse_selection_count_b: int,
    field_speeds: list[float],
    field_win_rates: list[float],
    field_size: int,
) -> dict[str, float]:
    """
    為單匹馬提取所有相對強度特徵。

    返回:
      {
        'vs_field_avg_speed': float,
        'vs_field_win_rate': float,
        'field_dominance_score': float,
        'upset_potential': float,
        'favorite_indicator': float,
      }
    """
    speed_adv = calc_vs_field_avg_speed(horse_speed, field_speeds)
    wr_adv = calc_vs_field_win_rate(horse_win_rate, field_win_rates)
    field_avg_speed = sum(field_speeds) / len(field_speeds) if field_speeds else horse_speed
    field_avg_class = sum(range(1, field_size + 1)) / field_size  # 簡化估計

    dominance = calc_field_dominance_score(speed_adv, wr_adv, horse_recent_form)
    upset_pot = calc_upset_potential(horse_class_rank, field_avg_class, horse_speed, field_avg_speed)
    favorite = calc_favorite_indicator(horse_selection_count_a, horse_selection_count_b, field_size)

    return {
        'vs_field_avg_speed': speed_adv,
        'vs_field_win_rate': wr_adv,
        'field_dominance_score': dominance,
        'upset_potential': upset_pot,
        'favorite_indicator': favorite,
    }
