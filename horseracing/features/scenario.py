"""
Scenario Features — 編碼場景特化訊號

分析發現：
  - Sha Tin: A & B 表現相當 (60.6%)
  - Happy Valley: A 明顯更優 (63.0% vs 47.2%, +15.8pp)

本模塊捕捉場地對模型選擇的影響。

特徵列表:
  - venue_model_alignment: 該場地對模型選擇的影響程度
  - field_strength_indicator: 該場馬匹整體等級（高手雲集 vs 菜雞互啄）
  - expected_uncertainty: 該場比賽的不確定程度
  - is_sha_tin: 是否為沙田 (one-hot)
  - is_happy_valley: 是否為快活谷 (one-hot)
"""

from __future__ import annotations


def calc_venue_model_alignment(
    venue: str,
    a_picks: list[str],
    b_picks: list[str],
) -> float:
    """
    計算該場地對 A & B 模型選擇的「一致性」。

    邏輯:
      - Sha Tin: 兩模型表現相當 → alignment = 0.5 (中立)
      - Happy Valley: A 更看好 → alignment = 0.7 (A 偏向)

    返回:
      float [0-1] 場地對模型的驅動力（越高越表示該場地有明確的偏好）
    """
    if venue.upper() == 'ST':
        return 0.5  # Sha Tin: 兩模型均衡
    elif venue.upper() == 'HV':
        return 0.7  # Happy Valley: 傾向 A
    else:
        return 0.5  # 未知場地，中性


def calc_field_strength_indicator(
    field_size: int,
    horses_history_avg: list[float],
) -> float:
    """
    計算該場馬匹的整體等級。

    邏輯:
      - 馬匹數越多（14 vs 12）→ 競爭越激烈
      - 馬匹歷史成績越好 → 等級越高

    參數:
      field_size: 該場馬匹數量 (通常 10-14)
      horses_history_avg: 每匹馬的歷史平均速度或勝率

    返回:
      float [0-1]
        0.2-0.4: 「菜雞互啄」，冷門多
        0.4-0.6: 中等水準
        0.7-1.0: 「高手雲集」，大熱馬容易贏
    """
    if not horses_history_avg or len(horses_history_avg) == 0:
        return 0.5

    avg_ability = sum(horses_history_avg) / len(horses_history_avg)

    # 基於平均能力 [0-1] 和場地大小 [10-14]
    size_factor = (field_size - 10) / 4.0  # [0-1]
    strength = avg_ability * 0.6 + size_factor * 0.4

    return min(1.0, max(0.0, strength))


def calc_expected_uncertainty(
    field_strength: float,
    horses_variance: float,
) -> float:
    """
    計算該場比賽的不確定程度（熵）。

    邏輯:
      - 高手雲集（field_strength 高）→ 結果可預測 → uncertainty 低
      - 菜雞互啄（field_strength 低）→ 結果難預測 → uncertainty 高
      - 馬匹能力差異大 → uncertainty 高
      - 馬匹能力相近 → uncertainty 高（任何馬都可能贏）

    返回:
      float [0-1] 不確定程度
    """
    # 高等級馬匹更容易預測（強者通常贏）
    predictability_from_strength = field_strength * 0.7

    # 馬匹差異大時（方差大），結果容易預測（因為有明確的強弱）
    predictability_from_variance = max(0.0, horses_variance - 0.3) * 0.3

    predictability = predictability_from_strength + predictability_from_variance
    uncertainty = 1.0 - predictability

    return min(1.0, max(0.0, uncertainty))


def extract_scenario_features(
    venue: str,
    field_size: int,
    horses_history_avg: list[float],
    a_picks: list[str],
    b_picks: list[str],
) -> dict[str, float]:
    """
    為單場賽事提取所有場景特化特徵。

    返回:
      {
        'venue_model_alignment': float,
        'field_strength_indicator': float,
        'expected_uncertainty': float,
        'is_sha_tin': int (0/1),
        'is_happy_valley': int (0/1),
      }
    """
    alignment = calc_venue_model_alignment(venue, a_picks, b_picks)

    # 計算馬匹能力方差
    if horses_history_avg and len(horses_history_avg) > 1:
        avg = sum(horses_history_avg) / len(horses_history_avg)
        variance = sum((x - avg) ** 2 for x in horses_history_avg) / len(horses_history_avg)
    else:
        variance = 0.1

    field_strength = calc_field_strength_indicator(field_size, horses_history_avg)
    uncertainty = calc_expected_uncertainty(field_strength, variance)

    is_st = 1 if venue.upper() == 'ST' else 0
    is_hv = 1 if venue.upper() == 'HV' else 0

    return {
        'venue_model_alignment': alignment,
        'field_strength_indicator': field_strength,
        'expected_uncertainty': uncertainty,
        'is_sha_tin': is_st,
        'is_happy_valley': is_hv,
    }
