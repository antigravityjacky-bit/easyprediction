# 馬匹預測系統重構計劃 - 基於 A/B 模型分析

**版本**: v1.0  
**狀態**: 實施準備中  
**目標**: Precision@3 從 20% → 60%+ (3倍改進)

---

## 📊 A/B 分析結果（80 場賽事）

### 關鍵發現

#### 1. 模型性能對比 ⭐

```
Model A (當前系統)
  - Precision@3: 61.7%
  - Hit Rate: 98.8%
  - 失敗場次: 1/80 (1.2%)

Model B (基線模型)
  - Precision@3: 54.6%
  - Hit Rate: 96.2%
  - 失敗場次: 3/80 (3.8%)

📈 目標系統應接近 Model A 的 62% 水準
```

#### 2. 超強信號：A & B 共識 🤝

```
A & B 同時選中的馬：
  - 出現次數: 225 馬次
  - 命中率: 48.4% ⭐⭐⭐ 超強信號!

相比之下：
  - 只有 A 選: 41.1% 命中率
  - 只有 B 選: 23.2% 命中率

💡 結論: 「雙模型共識」是最強訊號，應給予大幅權重提升
```

#### 3. 場地特化表現差異 🏁

```
Sha Tin (44 場):
  Model A: 60.6%
  Model B: 60.6%
  → 兩模型表現相當

Happy Valley (36 場):
  Model A: 63.0%
  Model B: 47.2%
  → A 有明顯優勢 (+15.8pp)

💡 結論: 場地差異是關鍵，需要場景特化特徵
```

#### 4. 模型互補性 🔄

```
失敗互補性：
  - A 失敗時 B 成功: 100%
  - B 失敗時 A 成功: 100%
  - 兩者都失敗: 0%

💡 結論: A & B 差異很大，組合可達最優效果
```

#### 5. 選擇位置分析 📍

```
A 模型：
  位置 1: 42.5% (首選保守)
  位置 2: 45.0%
  位置 3: 53.8% 🏆 (最有信心)
  位置 4: 43.8%

B 模型：
  位置 1: 50.0% 🏆 (首選有信心)
  位置 2: 48.8%
  位置 3: 30.0% (備選較弱)
  位置 4: 35.0%

💡 結論: A 和 B 的選擇邏輯完全不同
         - A: 漸進式置信（3 > 2 > 1）
         - B: 遞減式置信（1 > 2 > 3）
```

---

## 🎯 改造方案概覽

### Phase 1: 特徵工程（1-2 週）

新增 **50+ 特徵** 來編碼 A/B 分析發現的訊號

#### A. 共識特徵（強度最高）
```python
# horseracing/features/consensus.py [新文件]

1. agreement_signal
   - 基於 A & B 同時選中的歷史勝率
   - 計算過去 N 場賽事中，兩模型同時看好該馬的命中率
   - 權重: 最高 (+25% 提升空間)

2. agreement_strength
   - 衡量「共識強度」: 0-1 分數
   - 若馬匹在 A 選位置 1-2 + B 選位置 1-2 → 最高分
   - 若馬匹只在某模型位置 4 → 最低分

3. divergence_factor
   - 測量 A & B 的分歧程度
   - A 選該馬但 B 不選 → 輕微分歧
   - A & B 都不選但該馬贏了 → 大分歧（黑馬信號）
```

#### B. 選擇信心度特徵
```python
# horseracing/features/confidence.py [新文件]

1. A_selection_position
   - 馬匹在 A 選擇中的位置: 1, 2, 3, 4
   - 編碼為: [1→0.8, 2→0.75, 3→0.9, 4→0.6]
   - 反映 A 的選擇邏輯（位置 3 最有信心）

2. B_selection_position
   - 馬匹在 B 選擇中的位置: 1, 2, 3, 4
   - 編碼為: [1→0.95, 2→0.85, 3→0.4, 4→0.5]
   - 反映 B 的選擇邏輯（位置 1 最有信心）

3. combined_confidence
   - 結合 A & B 的置信度加權平均
   - 公式: 0.6 * A_conf + 0.4 * B_conf（基於 A 更準）

4. confidence_agreement
   - 若 A & B 都給出高置信度 → 最高分
   - 若 A & B 置信度不一致 → 降分

5. banker_signal
   - 若馬匹是 A 的位置 1 或 2 → Banker 機率高
   - 编码馬匹作為主注的概率
```

#### C. 場景特化特徵
```python
# horseracing/features/scenario.py [新文件]

1. venue_model_alignment
   - Sha Tin: A & B 同等看好 → alignment_score = 0.5
   - Happy Valley: A 更看好 → alignment_score = 0.6-0.8
   - 編碼場地對模型選擇的影響

2. venue_performance_record
   - 該馬在 Sha Tin vs Happy Valley 的歷史勝率
   - 場地特化能力指標

3. field_strength_indicator
   - 根據場次馬匹數量和平均能力級別
   - 預測該場是否為「高手雲集」或「菜雞互啄」
   - 影響冷門概率

4. competition_density
   - 該馬與同場其他馬的相對強度
   - 如果同場全是弱馬，該馬選中的可靠性更高
```

#### D. 衰減歷史特徵
```python
# horseracing/features/recency.py [新文件]

1. recent_3races_win_rate (衰減)
   - 最近 3 場的成績加權平均（decay=0.7, 0.5, 1.0）
   - 捕捉「狀態好壞」比長期平均更重要

2. recent_6races_trend
   - 過去 6 場的趨勢線：上升 / 平穩 / 下降
   - 編碼為: [+0.2, 0, -0.2]

3. form_momentum
   - 最近 N 場的成績加速度（二階導數）
   - 如果連贏或連敗，信號強度增大

4. layoff_penalty
   - days_since_last 的非線性懲罰
   - 超過 60 天未跑 → 大幅懲罰
   - 7-14 天未跑 → 小幅獎勵（狀態調整）
```

#### E. 馬匹-騎師配對特徵
```python
# horseracing/features/pairing.py [新文件]

1. jockey_horse_affinity
   - 特定騎師與特定馬的歷史成績比率
   - 該騎師在該馬上的命中率

2. jockey_distance_affinity
   - 騎師在該距離的成績（區別於馬的距離親和）

3. recent_jockey_form
   - 騎師最近 10 場的成績加權趨勢
   - 騎師狀態可能直接影響預測

4. jockey_A_B_agreement
   - 如果同一騎師出現在 A & B 都選中的馬上 → 額外加分
   - 騎師與馬的搭配可能是隱藏信號
```

#### F. 相對強度特徵
```python
# horseracing/features/relative_strength.py [新文件]

1. vs_field_avg_speed
   - 該馬的歷史平均速度 vs 同場馬匹平均
   - 相對優勢指標

2. vs_field_win_rate
   - 該馬的歷史勝率 vs 同場其他馬的平均
   - 場內統治力

3. upset_potential
   - 如果馬匹的等級低於同場平均但能力強 → 黑馬信號
   - 編碼被低估的馬

4. favorite_vs_outsider
   - 基於選擇頻率估計「大熱馬」vs「冷馬」
   - 如果冷馬被 A & B 都選中 → 信號更強
```

### Phase 2: 訓練策略改進（1-2 週）

#### A. 特徵矩陣重建
```python
# 修改: horseracing/ml/dataset.py

def build_dataset(include_new_features=True):
    """
    重建 ml_dataset.csv，加入 50+ 新特徵
    
    新特徵來源:
    1. consensus.py: agreement_signal, agreement_strength, divergence_factor
    2. confidence.py: A/B selection position, combined_confidence
    3. scenario.py: venue alignment, field strength
    4. recency.py: recent forms, momentum
    5. pairing.py: jockey affinity
    6. relative_strength.py: vs field metrics
    
    總特徵數: 52 → 120+
    """
```

#### B. 訓練目標改進
```python
# 修改: horseracing/ml/models.py

# 舊方式：簡單二分類
# y_train = (finish_position <= 3) → 0/1

# 新方式：多任務學習
class MultiTaskLearner:
    """
    Task 1: P(top-3)  [保留]
      - 傳統二分類，預測馬是否在前三名
      - 權重: 0.4
    
    Task 2: ranking_probability
      - 預測馬匹的相對排序（1st, 2nd, 3rd, 4th+ 的概率分佈）
      - 權重: 0.4
      - 利用 finish_position 的相對信息
    
    Task 3: agreement_correction
      - 預測 A & B 共識是否為真（0/1）
      - 權重: 0.2
      - 當 A & B 都選該馬時，是否真的會贏
      - 這是一個「可信度」任務
    """
```

#### C. 模型超參優化
```python
# 修改: horseracing/ml/train.py

# LightGBM 配置調整
lgbm_params_new = {
    "objective": "multi_logloss",  # 改為多分類（1/2/3/4+）
    "num_classes": 4,
    "metric": "multi_logloss",
    "num_leaves": 127,  # 增加複雜度
    "learning_rate": 0.02,
    "n_estimators": 1200,  # 增加迭代
    "min_child_samples": 3,  # 降低最小樣本，增加適應性
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,  # 增加正則化
    "reg_lambda": 1.0,
    "max_depth": 10,
    "class_weight": {0: 1, 1: 3, 2: 2.5, 3: 2},  # 不同排名的權重
}

# 使用 early_stopping 防止過擬合
```

### Phase 3: 集成策略改進（1 週）

#### A. 動態權重分配
```python
# 修改: horseracing/ml/ensemble.py

def smart_ensemble(
    horse_idx, 
    race_venue,
    model_A_predicts,
    model_B_predicts,
    **model_outputs
):
    """
    根據場景動態調整權重：
    
    Scenario 1: Sha Tin + A & B 高度共識
      weights = {agreement: 0.7, lgbm: 0.15, ltr: 0.15}
    
    Scenario 2: Happy Valley + A 獨有選擇
      weights = {A_unique: 0.5, lgbm: 0.3, ltr: 0.2}
    
    Scenario 3: 高度不確定（A & B 分歧大）
      weights = {ensemble: 0.7, conservative: 0.3}
    """
```

#### B. 後處理規則
```python
# 修改: horseracing/prediction/live_predictor.py

def post_process_predictions(raw_probs, horse_info, context):
    """
    應用啟發式規則優化預測：
    
    Rule 1: 共識信號增強
      if horse in A_picks and horse in B_picks:
          prob *= 1.3  # 提升 30%
    
    Rule 2: 高信心位置優先
      if horse == A_position_3 or horse == B_position_1:
          prob *= 1.2
    
    Rule 3: 黑馬懲罰
      if horse == outlaw_horse and not_in_A_B_consensus:
          prob *= 0.7
    
    Rule 4: 馬匹等級調整
      if horse.class > field_avg:
          prob *= 1.1  # 高等級馬更可能贏
    """
```

---

## 📝 實施清單

### Week 1: 特徵工程

- [ ] **新建特徵模塊**
  - [ ] `horseracing/features/consensus.py` (agreement_signal, etc.)
  - [ ] `horseracing/features/confidence.py` (selection position)
  - [ ] `horseracing/features/scenario.py` (venue, field strength)
  - [ ] `horseracing/features/recency.py` (衰減歷史)
  - [ ] `horseracing/features/pairing.py` (騎師配對)
  - [ ] `horseracing/features/relative_strength.py` (相對強度)

- [ ] **更新特徵列表**
  - [ ] 修改 `horseracing/features/engineer.py` 中的 `FEATURE_NAMES`
  - [ ] 從 52 個特徵擴展到 120+ 個

- [ ] **重編譯數據集**
  - [ ] 修改 `horseracing/ml/dataset.py` 中的 `build_dataset()`
  - [ ] 重新編譯 `datasets/processed/ml_dataset.csv`

### Week 2: 訓練策略改進

- [ ] **改進模型訓練**
  - [ ] 修改 `horseracing/ml/models.py` 支援多任務學習
  - [ ] 更新超參配置（num_leaves, learning_rate 等）
  - [ ] 實現多任務損失函數

- [ ] **重新訓練模型**
  - [ ] 執行 `python -m horseracing.ml.train`
  - [ ] 評估新模型在舊測試集上的表現

- [ ] **驗證回測**
  - [ ] 執行 `python -m horseracing.backtest.runner`
  - [ ] 目標: Precision@3 達到 50%+ (從當前 20%)

### Week 3: 集成優化

- [ ] **動態集成策略**
  - [ ] 修改 `horseracing/ml/ensemble.py` 實現智能加權
  - [ ] 根據場景（venue, agreement level）調整權重

- [ ] **後處理規則**
  - [ ] 修改 `horseracing/prediction/live_predictor.py`
  - [ ] 實現共識增強、黑馬懲罰等規則

- [ ] **最終驗證**
  - [ ] 完整回測（目標 60%+ Precision@3）
  - [ ] 對比舊系統（20%）和 Model A（62%）

### Week 4-5: 測試迭代

- [ ] **深度分析**
  - [ ] 分析失敗案例，識別新訊號
  - [ ] 調整特徵權重和模型參數

- [ ] **部署準備**
  - [ ] 更新 `ensemble_weights.json`
  - [ ] 保存新訓練的模型
  - [ ] 編寫改造文檔

---

## 🎯 預期成果

| 指標 | 當前系統 | 改造後 | 達成路徑 |
|-----|--------|-------|--------|
| **Precision@3** | 20% | 60%+ 🎯 | 特徵工程 + 多任務學習 |
| **Hit Rate** | 低 | 95%+ | 共識信號 + 智能集成 |
| **場地適應** | 統一模型 | 場景特化 | venue-based sub-models |
| **模型互補** | 權重固定 | 動態加權 | 場景感知集成 |

---

## 🔧 關鍵檔案修改清單

```
horseracing/
├── features/
│   ├── engineer.py              [改] FEATURE_NAMES 擴展 52→120+
│   ├── consensus.py             [新] 共識特徵
│   ├── confidence.py            [新] 置信度特徵
│   ├── scenario.py              [新] 場景特化特徵
│   ├── recency.py               [新] 衰減歷史特徵
│   ├── pairing.py               [新] 馬匹-騎師特徵
│   └── relative_strength.py     [新] 相對強度特徵
│
├── ml/
│   ├── dataset.py               [改] build_dataset() 加新特徵
│   ├── models.py                [改] 多任務學習支援
│   ├── ensemble.py              [改] 動態權重分配
│   └── train.py                 [改] 訓練配置、超參優化
│
├── prediction/
│   ├── live_predictor.py        [改] 後處理規則、WEIGHTS 更新
│   └── engine.py                [無改] 自動獲得新特徵
│
└── backtest/
    ├── runner.py                [無改] 自動適應新特徵
    └── analyzer.py              [無改] 自動評估新模型

datasets/
├── processed/
│   ├── ml_dataset.csv           [重編] 包含新特徵
│   └── models/
│       ├── lgbm_top3.pkl        [重訓] 新模型
│       ├── xgb_top3.pkl         [重訓] 新模型
│       ├── ltr_ranker.pkl       [無改]
│       └── ensemble_weights.json [更新]
```

---

## ✅ 驗證標準

**成功條件** (需全部達成):

1. ✅ **Precision@3 ≥ 50%** (中期目標)
   - 回測數據上，預測的 Top-3 中至少 50% 準確

2. ✅ **Hit Rate ≥ 90%** (覆蓋率)
   - 至少 90% 的場次預測中至少一匹馬

3. ✅ **對比基線** (v.s. Model A)
   - 新系統 Precision@3 ≥ 60% (Model A 水準)

4. ✅ **場地適應** (場景穩定性)
   - Sha Tin 和 Happy Valley 的表現差異 < 5pp

5. ✅ **無資料洩漏** (時間序列完整性)
   - 所有特徵只使用該賽事前的歷史數據
   - 驗證 CV 分割無重疊

---

## 🚀 開始實施

### 即刻可做:

1. 在 `/home/user/easyprediction/claude-refactor` 分支開始開發
2. 新建 Phase 1 的特徵模塊
3. 驗證新特徵的可計算性

### 依賴項:

- 現有訓練數據: `datasets/processed/ml_dataset.csv` ✅
- 現有模型架構: `ml/models.py` ✅
- RollingTracker: `ml/dataset.py` ✅

---

**下一步**: 開始 Phase 1 特徵工程實施 → 目標 2 週完成

