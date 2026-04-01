"""
Consensus Features — 編碼 A & B 模型共識訊號

基於 A/B 模型對比分析，「A & B 同時選中的馬」具有最強預測力（命中率 48.4%）。
本模塊設計特徵來捕捉這一超強訊號。

特徵列表:
  - agreement_signal: 馬匹在 A & B 都看好時的歷史命中率
  - agreement_strength: 衡量共識強度 (0-1)
  - divergence_factor: A & B 分歧程度（0 = 完全共識，1 = 完全分歧）
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict


def calc_agreement_signal(
    horse_code: str,
    history: list[dict],
    window_races: int = 20,
) -> float:
    """
    計算該馬「A & B 同時看好時的命中率」。

    參數:
      horse_code: 馬匹代碼
      history: [{date, A_picks, B_picks, actual_top3, ...}] 歷史賽事列表
      window_races: 回看窗口 (最近 N 場)

    返回:
      float [0-1] 該馬在 A & B 同時選中時的命中率

    邏輯:
      1. 篩選過去 N 場賽事中該馬出現的次數
      2. 在這些次數中，找出 A & B 都選中的場次
      3. 在 A & B 都選中的場次中，計算有多少次該馬進入前三名
      4. 返回: (命中次數 / A & B 都選次數)
    """
    if not history or len(history) == 0:
        return 0.5  # 中性信號，無歷史數據時

    # 限制窗口
    recent = history[-window_races:] if len(history) > window_races else history

    both_selected_count = 0
    both_selected_and_hit = 0

    for race in recent:
        if horse_code not in race.get('all_horses', []):
            continue

        # 檢查 A & B 是否都選中該馬
        a_picks = set(race.get('A_picks', []))
        b_picks = set(race.get('B_picks', []))

        if horse_code in a_picks and horse_code in b_picks:
            both_selected_count += 1
            # 檢查該馬是否進入前三名
            actual_top3 = set(race.get('actual_top3', []))
            if horse_code in actual_top3:
                both_selected_and_hit += 1

    if both_selected_count == 0:
        return 0.5  # 無共識歷史時

    hit_rate = both_selected_and_hit / both_selected_count
    # 防止過度擬合，使用拉普拉斯平滑
    return (both_selected_and_hit + 1) / (both_selected_count + 2)


def calc_agreement_strength(
    horse_code: str,
    a_position: int | None,  # 1, 2, 3, 4 或 None (未選中)
    b_position: int | None,  # 同上
) -> float:
    """
    計算「共識強度」分數，反映 A & B 對該馬的置信度。

    參數:
      horse_code: 馬匹代碼
      a_position: A 模型中的選擇位置 (1=首選, 2=次選, 3=三選, 4=備選, None=未選)
      b_position: B 模型中的選擇位置

    返回:
      float [0-1] 共識強度分數

    邏輯:
      - 都未選: 0.0
      - 只有一者選: 0.3-0.5 (輕微共識)
      - 都選但位置差異大: 0.6 (中等共識)
      - 都選且位置近: 0.8-1.0 (強共識)
    """
    if a_position is None and b_position is None:
        return 0.0

    if a_position is None or b_position is None:
        # 只有一者選中
        return 0.35

    # 兩者都選中 — 計算位置接近度
    # A: 位置 3 最有信心 (A_confidence[pos] = [0.8, 0.75, 0.9, 0.6])
    # B: 位置 1 最有信心 (B_confidence[pos] = [0.95, 0.85, 0.4, 0.5])

    a_conf = [0.80, 0.75, 0.90, 0.60][a_position - 1]
    b_conf = [0.95, 0.85, 0.40, 0.50][b_position - 1]

    # 位置接近度：位置差距越小，接近度越高
    position_proximity = 1.0 - (abs(a_position - b_position) / 3.0)  # [0-1]

    # 最終共識強度 = (A 置信 + B 置信) / 2 * 位置接近度
    consensus = ((a_conf + b_conf) / 2.0) * (0.5 + 0.5 * position_proximity)

    return min(1.0, consensus)


def calc_divergence_factor(
    horse_code: str,
    a_picks: set[str],
    b_picks: set[str],
    actual_top3: set[str],
) -> float:
    """
    計算 A & B 的「分歧因子」。

    參數:
      horse_code: 馬匹代碼
      a_picks: A 模型的選擇集合
      b_picks: B 模型的選擇集合
      actual_top3: 實際前三名馬匹

    返回:
      float [-1, 1]
        -1: 完全共識（A & B 都選且贏了，或都沒選且沒贏）
        0: 中立（一致認為不會贏）
        +1: 完全分歧（A 選 B 不選，或 B 選 A 不選，且該馬贏了）
    """
    in_a = horse_code in a_picks
    in_b = horse_code in b_picks
    won = horse_code in actual_top3

    # 完全共識情況
    if (in_a and in_b and won) or (not in_a and not in_b and not won):
        return -1.0

    # 部分分歧：只有一方選中
    if (in_a or in_b) and not won:
        return 0.3  # 分歧較小，因為都沒贏

    if (in_a and not in_b) or (not in_a and in_b):
        if won:
            return 0.8  # 一方獨有正確預測，分歧強
        else:
            return 0.2  # 一方看走眼，分歧較小

    return 0.0


def extract_consensus_features(
    horse_code: str,
    a_picks: list[str],
    b_picks: list[str],
    history: list[dict],
) -> dict[str, float]:
    """
    為單匹馬提取所有共識特徵。

    返回:
      {
        'agreement_signal': float,
        'agreement_strength': float,
        'divergence_factor': float,
        'is_consensus_pick': int (0/1),
      }
    """
    a_picks_set = set(a_picks)
    b_picks_set = set(b_picks)

    # 計算 A & B 中的位置
    a_position = None
    b_position = None

    if horse_code in a_picks:
        a_position = a_picks.index(horse_code) + 1
    if horse_code in b_picks:
        b_position = b_picks.index(horse_code) + 1

    agreement_signal = calc_agreement_signal(horse_code, history)
    agreement_strength = calc_agreement_strength(horse_code, a_position, b_position)

    # divergence_factor 在實際使用時會用實際結果計算
    # 這裡先預留 0 值，回測時會更新
    divergence_factor = 0.0

    # 是否為共識選擇（都在前 3 選中）
    is_consensus = 1 if (a_position is not None and a_position <= 3 and
                          b_position is not None and b_position <= 3) else 0

    return {
        'agreement_signal': agreement_signal,
        'agreement_strength': agreement_strength,
        'divergence_factor': divergence_factor,
        'is_consensus_pick': is_consensus,
    }
