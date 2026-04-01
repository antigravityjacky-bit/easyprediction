"""
Confidence Features — 編碼 A & B 模型的選擇置信度

A 和 B 模型在不同位置上的置信度完全不同：
  - A 模型：位置 3 最有信心（53.8% 命中率）→ 漸進式置信
  - B 模型：位置 1 最有信心（50.0% 命中率）→ 遞減式置信

本模塊捕捉這一差異，為 ML 模型提供「選擇位置信息」。

特徵列表:
  - a_selection_position: A 選擇的位置信號 (1-4 或 None)
  - b_selection_position: B 選擇的位置信號 (1-4 或 None)
  - combined_confidence: A & B 置信度的加權平均
  - confidence_agreement: A & B 置信度是否一致
  - banker_signal: 馬匹作為「Banker（主注）」的概率
"""

from __future__ import annotations


# A 模型在各位置的命中率（實測）
A_POSITION_HITRATES = {
    1: 0.425,  # 首選
    2: 0.450,  # 次選
    3: 0.538,  # 三選（最有信心！）
    4: 0.438,  # 備選
}

# B 模型在各位置的命中率（實測）
B_POSITION_HITRATES = {
    1: 0.500,  # 首選（最有信心！）
    2: 0.488,  # 次選
    3: 0.300,  # 三選（較弱）
    4: 0.350,  # 備選
}

# 將命中率轉為置信度分數 [0-1]
A_POSITION_CONFIDENCE = {pos: rate for pos, rate in A_POSITION_HITRATES.items()}
B_POSITION_CONFIDENCE = {pos: rate for pos, rate in B_POSITION_HITRATES.items()}


def encode_a_selection_position(a_position: int | None) -> float:
    """
    編碼 A 模型的選擇位置為置信度分數。

    參數:
      a_position: A 選擇的位置 (1, 2, 3, 4) 或 None

    返回:
      float [0-1] A 的置信度分數
    """
    if a_position is None:
        return 0.0

    return A_POSITION_CONFIDENCE.get(a_position, 0.0)


def encode_b_selection_position(b_position: int | None) -> float:
    """
    編碼 B 模型的選擇位置為置信度分數。

    參數:
      b_position: B 選擇的位置 (1, 2, 3, 4) 或 None

    返回:
      float [0-1] B 的置信度分數
    """
    if b_position is None:
        return 0.0

    return B_POSITION_CONFIDENCE.get(b_position, 0.0)


def calc_combined_confidence(
    a_position: int | None,
    b_position: int | None,
    a_weight: float = 0.60,
    b_weight: float = 0.40,
) -> float:
    """
    結合 A & B 的置信度為單一分數。

    參數:
      a_position: A 的選擇位置
      b_position: B 的選擇位置
      a_weight: A 的權重（A 更準）
      b_weight: B 的權重

    返回:
      float [0-1] 組合置信度
    """
    a_conf = encode_a_selection_position(a_position)
    b_conf = encode_b_selection_position(b_position)

    # 若只有一者選中，給予輕微降分
    if (a_position is None or b_position is None) and not (a_position is None and b_position is None):
        penalty = 0.1
    else:
        penalty = 0.0

    combined = a_weight * a_conf + b_weight * b_conf - penalty
    return max(0.0, min(1.0, combined))


def calc_confidence_agreement(
    a_position: int | None,
    b_position: int | None,
) -> float:
    """
    計算 A & B 置信度是否一致 (alignment)。

    返回:
      float [0-1]
        1.0: 完全一致（都是高置信位置，如 A_pos3 + B_pos1）
        0.5: 部分一致（一高一低）
        0.0: 完全不一致（都未選，或一選一不選）
    """
    if a_position is None and b_position is None:
        return 0.0

    if a_position is None or b_position is None:
        return 0.2  # 輕微不一致

    a_conf = A_POSITION_CONFIDENCE[a_position]
    b_conf = B_POSITION_CONFIDENCE[b_position]

    # 置信度接近度：絕對差距越小，一致性越高
    max_conf = max(a_conf, b_conf)
    min_conf = min(a_conf, b_conf)

    if max_conf < 0.4:
        return 0.1  # 都不太看好

    alignment = min_conf / max_conf  # 相對比例 [0-1]
    return alignment


def calc_banker_signal(
    a_position: int | None,
    b_position: int | None,
) -> float:
    """
    計算馬匹作為「Banker（主注）」的概率。

    Banker 是指最有把握的馬匹（通常是第一選擇）。

    邏輯:
      - A_position <= 2 and B_position <= 2: 高概率 Banker (~0.8-0.9)
      - A_position == 3 and B_position <= 1: 中等概率 (~0.6-0.7)
      - 其他: 低概率 (~0.2-0.4)
    """
    if a_position is None or b_position is None:
        return 0.0

    # 都在前 2 選
    if a_position <= 2 and b_position <= 2:
        return 0.85

    # A position 3（特殊情況，A 在此位置信心最高）
    if a_position == 3 and b_position <= 1:
        return 0.75

    # 一方首選，一方次選
    if (a_position <= 2 and b_position == 2) or (b_position <= 2 and a_position == 2):
        return 0.65

    # 其他情況
    return 0.25


def extract_confidence_features(
    horse_code: str,
    a_picks: list[str],
    b_picks: list[str],
) -> dict[str, float]:
    """
    為單匹馬提取所有置信度特徵。

    返回:
      {
        'a_selection_position': float,
        'b_selection_position': float,
        'combined_confidence': float,
        'confidence_agreement': float,
        'banker_signal': float,
      }
    """
    # 計算位置
    a_position = None
    b_position = None

    if horse_code in a_picks:
        a_position = a_picks.index(horse_code) + 1
    if horse_code in b_picks:
        b_position = b_picks.index(horse_code) + 1

    # 提取特徵
    a_conf = encode_a_selection_position(a_position)
    b_conf = encode_b_selection_position(b_position)
    combined_conf = calc_combined_confidence(a_position, b_position)
    conf_agreement = calc_confidence_agreement(a_position, b_position)
    banker_sig = calc_banker_signal(a_position, b_position)

    return {
        'a_selection_position': a_conf,
        'b_selection_position': b_conf,
        'combined_confidence': combined_conf,
        'confidence_agreement': conf_agreement,
        'banker_signal': banker_sig,
    }
