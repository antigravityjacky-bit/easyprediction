#!/usr/bin/env python3
"""
信心場次 - 實際Pattern分析
分析成功場次(>=2中)嘅實際特徵:
  1. Race number pattern (R1, R2, ... R11邊個最成功?)
  2. 馬號pattern (共識馬都係邊啲號碼?)
  3. 場次特徵 (venue, handicap, race number combination)
"""

import csv
from collections import defaultdict, Counter

# Load March data
march_data = []
with open('/home/user/easyprediction/datasets/processed/march_2026_comprehensive_dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        march_data.append(row)

# Test dates - 6 days
target_dates = ['2026-03-04', '2026-03-08', '2026-03-11', '2026-03-15', '2026-03-18', '2026-03-25']

# Handicap markers
handicap_markers = {
    '2026-03-04': True,
    '2026-03-08': False,
    '2026-03-11': True,
    '2026-03-15': False,
    '2026-03-18': True,
    '2026-03-25': True,
}

def method_y_evolved(a_picks, b_picks):
    """Get consensus horses"""
    a_set = set(a_picks)
    b_set = set(b_picks)
    return list(a_set & b_set)

# Analyze success/failure cases
success_cases = []
failure_cases = []

for row in march_data:
    if row['Date'] not in target_dates:
        continue

    date = row['Date']
    venue = row['Venue']
    race_no = row['Race_No']
    is_handicap = handicap_markers.get(date, False)

    a_picks = [row['ModelA_Pick_1'], row['ModelA_Pick_2'], row['ModelA_Pick_3'], row['ModelA_Pick_4']]
    b_picks = [row['ModelB_Pick_1'], row['ModelB_Pick_2'], row['ModelB_Pick_3'], row['ModelB_Pick_4']]
    actual = [row['Actual_Top3_1'], row['Actual_Top3_2'], row['Actual_Top3_3']]

    consensus = method_y_evolved(a_picks, b_picks)
    pred_y = sorted(consensus + list(set(a_picks) - set(consensus)) + list(set(b_picks) - set(consensus)))[:3]

    hits = len([h for h in pred_y if h in actual])

    case = {
        'date': date,
        'venue': venue,
        'race_no': race_no,
        'is_handicap': is_handicap,
        'hits': hits,
        'consensus': sorted([int(h) for h in consensus]),
        'actual': sorted([int(h) for h in actual]),
        'prediction': pred_y,
    }

    if hits >= 2:
        success_cases.append(case)
    else:
        failure_cases.append(case)

print("=" * 100)
print("實際Pattern分析 - 成功vs失敗場次")
print("=" * 100)
print(f"\n總樣本: {len(success_cases) + len(failure_cases)} 場")
print(f"成功: {len(success_cases)} 場 (≥2中)")
print(f"失敗: {len(failure_cases)} 場 (<2中)")

# ============================================
# 1. Race Number Pattern
# ============================================
print("\n" + "=" * 100)
print("1️⃣ RACE NUMBER PATTERN - 邊個Race最容易中?")
print("=" * 100)

success_race_dist = Counter([c['race_no'] for c in success_cases])
failure_race_dist = Counter([c['race_no'] for c in failure_cases])

print("\n成功場次按Race Number分佈:")
for race in sorted(success_race_dist.keys(), key=lambda x: int(x.replace('R',''))):
    count = success_race_dist[race]
    total = count + failure_race_dist.get(race, 0)
    success_rate = count / total * 100 if total > 0 else 0
    print(f"  {race:2}: {count:2} 成功 / {total:2} 總場 = {success_rate:5.1f}% ✅" if success_rate >= 55 else f"  {race:2}: {count:2} 成功 / {total:2} 總場 = {success_rate:5.1f}% ⚠️")

print("\n失敗場次按Race Number分佈:")
for race in sorted(failure_race_dist.keys(), key=lambda x: int(x.replace('R',''))):
    count = failure_race_dist[race]
    total = count + success_race_dist.get(race, 0)
    failure_rate = count / total * 100 if total > 0 else 0
    print(f"  {race:2}: {count:2} 失敗 / {total:2} 總場 = {failure_rate:5.1f}%")

# Find best race
best_race = max(success_race_dist.keys(), key=lambda r: success_race_dist[r] / (success_race_dist[r] + failure_race_dist.get(r, 0)))
best_success_count = success_race_dist[best_race]
best_total = best_success_count + failure_race_dist.get(best_race, 0)
best_rate = best_success_count / best_total * 100
print(f"\n🏆 最佳Race: {best_race} - {best_success_count}/{best_total} = {best_rate:.1f}% 成功率")

# ============================================
# 2. Horse Number Patterns in Consensus
# ============================================
print("\n" + "=" * 100)
print("2️⃣ 馬號PATTERN - 成功時共識馬係邊啲號碼?")
print("=" * 100)

# 2a. Individual horse number in consensus
print("\n成功場次:共識馬中出現嘅馬號 (頻率):")
success_horses = []
for case in success_cases:
    success_horses.extend(case['consensus'])
success_horse_counter = Counter(success_horses)

for horse in sorted(success_horse_counter.keys()):
    count = success_horse_counter[horse]
    print(f"  馬{horse:2}: 出現 {count:2} 次", end="")
    if count >= 5:
        print(" ⭐⭐")
    elif count >= 3:
        print(" ⭐")
    else:
        print()

print("\n失敗場次:共識馬中出現嘅馬號 (頻率):")
failure_horses = []
for case in failure_cases:
    failure_horses.extend(case['consensus'])
failure_horse_counter = Counter(failure_horses)

for horse in sorted(failure_horse_counter.keys()):
    count = failure_horse_counter[horse]
    total = count + success_horse_counter.get(horse, 0)
    success_rate = success_horse_counter.get(horse, 0) / total * 100 if total > 0 else 0
    print(f"  馬{horse:2}: 出現 {count:2} 次 | 整體成功率 {success_rate:5.1f}%")

# 2b. Horse number groups (起始馬號)
print("\n\n共識馬號組合類型分析:")
print("(e.g., [1,2,3], [4,5,6]等類似嘅起始馬號組合)")

# Categorize by first horse number
def categorize_by_first_horse(consensus):
    if not consensus:
        return "無共識"
    first = min(consensus)
    if first <= 3:
        return "早期馬(1-3號)"
    elif first <= 6:
        return "中期馬(4-6號)"
    else:
        return "後期馬(7+號)"

success_categories = Counter([categorize_by_first_horse(c['consensus']) for c in success_cases])
failure_categories = Counter([categorize_by_first_horse(c['consensus']) for c in failure_cases])

print("\n成功場次 - 按共識馬嘅起始號碼分類:")
for cat in sorted(success_categories.keys()):
    count = success_categories[cat]
    total = count + failure_categories.get(cat, 0)
    rate = count / total * 100 if total > 0 else 0
    print(f"  {cat:15}: {count:2} / {total:2} = {rate:5.1f}% 成功")

# ============================================
# 3. Specific Consensus Combinations
# ============================================
print("\n" + "=" * 100)
print("3️⃣ 共識馬號組合 - 邊啲組合最成功?")
print("=" * 100)

# Count specific consensus combinations
success_combo_counter = Counter()
failure_combo_counter = Counter()

for case in success_cases:
    combo = tuple(sorted(case['consensus']))
    success_combo_counter[combo] += 1

for case in failure_cases:
    combo = tuple(sorted(case['consensus']))
    failure_combo_counter[combo] += 1

print("\n成功場次中最常見嘅共識馬號組合:")
for combo, count in success_combo_counter.most_common(10):
    total = count + failure_combo_counter.get(combo, 0)
    rate = count / total * 100
    print(f"  {str(list(combo)):20} : {count:2} 成功 / {total:2} 總 = {rate:5.1f}% ⭐" if rate >= 60 else f"  {str(list(combo)):20} : {count:2} 成功 / {total:2} 總 = {rate:5.1f}%")

# ============================================
# 4. Consensus Count Analysis
# ============================================
print("\n" + "=" * 100)
print("4️⃣ 共識馬數量 - 有幾隻共識馬最成功?")
print("=" * 100)

success_count_dist = Counter([len(c['consensus']) for c in success_cases])
failure_count_dist = Counter([len(c['consensus']) for c in failure_cases])

print("\n共識馬數量分佈:")
for count in sorted(set(list(success_count_dist.keys()) + list(failure_count_dist.keys()))):
    success = success_count_dist.get(count, 0)
    failure = failure_count_dist.get(count, 0)
    total = success + failure
    rate = success / total * 100 if total > 0 else 0
    print(f"  {count} 隻共識馬: {success:2} 成功 / {failure:2} 失敗 = {rate:5.1f}% 成功率")

# ============================================
# 5. Venue + Race Type Combinations
# ============================================
print("\n" + "=" * 100)
print("5️⃣ 場地+比賽類型 - 邊個組合最成功?")
print("=" * 100)

venue_race_dist = defaultdict(lambda: {'success': 0, 'failure': 0})

for case in success_cases:
    race_type = "讓賽" if case['is_handicap'] else "普通賽"
    key = f"{case['venue']} {race_type}"
    venue_race_dist[key]['success'] += 1

for case in failure_cases:
    race_type = "讓賽" if case['is_handicap'] else "普通賽"
    key = f"{case['venue']} {race_type}"
    venue_race_dist[key]['failure'] += 1

print("\n場地+比賽類型成功率:")
for key in sorted(venue_race_dist.keys()):
    success = venue_race_dist[key]['success']
    failure = venue_race_dist[key]['failure']
    total = success + failure
    rate = success / total * 100 if total > 0 else 0
    print(f"  {key:15}: {success:2} / {total:2} = {rate:5.1f}% 成功率")

# ============================================
# 6. Actual Top 3 Pattern
# ============================================
print("\n" + "=" * 100)
print("6️⃣ 實際頭3馬號PATTERN - 成功時嘅特徵?")
print("=" * 100)

print("\n成功場次 - 實際頭3馬號出現頻率:")
success_actual_horses = []
for case in success_cases:
    success_actual_horses.extend(case['actual'])
success_actual_counter = Counter(success_actual_horses)

for horse in sorted(success_actual_counter.keys()):
    count = success_actual_counter[horse]
    print(f"  馬{horse:2}: {count:2} 次", end="")
    if count >= 5:
        print(" ⭐⭐ (常勝馬!)")
    else:
        print()

# ============================================
# 7. Detailed Success Cases by Pattern
# ============================================
print("\n" + "=" * 100)
print("7️⃣ 成功場次詳細 - 按共識馬號排序")
print("=" * 100)

success_by_combo = defaultdict(list)
for case in success_cases:
    combo = tuple(sorted(case['consensus']))
    success_by_combo[combo].append(case)

print("\n按共識馬號組合分類 (成功場次):")
for combo in sorted(success_by_combo.keys(), key=lambda x: len(success_by_combo[x]), reverse=True)[:5]:
    cases = success_by_combo[combo]
    print(f"\n共識馬 {list(combo)} - {len(cases)} 場成功:")
    for case in cases[:3]:  # Show first 3
        print(f"  {case['date']} {case['venue']} R{case['race_no']}: 實際{case['actual']}")
    if len(cases) > 3:
        print(f"  ... 還有 {len(cases)-3} 場")

# ============================================
# 8. Summary & Insights
# ============================================
print("\n" + "=" * 100)
print("📊 SUMMARY - 最可靠嘅信心場次Pattern")
print("=" * 100)

# Find most reliable patterns
print("\n✅ 最可靠嘅條件 (成功率>60%):")

for count in sorted(success_count_dist.keys()):
    success = success_count_dist.get(count, 0)
    failure = failure_count_dist.get(count, 0)
    total = success + failure
    rate = success / total * 100 if total > 0 else 0
    if rate >= 60:
        print(f"  • {count} 隻共識馬: {rate:.1f}% 成功")

for race in sorted(success_race_dist.keys(), key=lambda x: int(x.replace('R',''))):
    count = success_race_dist[race]
    total = count + failure_race_dist.get(race, 0)
    rate = count / total * 100 if total > 0 else 0
    if rate >= 60:
        print(f"  • {race} 場: {rate:.1f}% 成功")

for combo in sorted(success_combo_counter.keys(), key=lambda x: success_combo_counter[x], reverse=True)[:5]:
    count = success_combo_counter[combo]
    total = count + failure_combo_counter.get(combo, 0)
    rate = count / total * 100 if total > 0 else 0
    if rate >= 60 and total >= 2:
        print(f"  • 共識 {list(combo)}: {rate:.1f}% 成功 ({count}/{total}場)")

print("\n⚠️ 風險較高嘅條件 (成功率<50%):")

for count in sorted(success_count_dist.keys()):
    success = success_count_dist.get(count, 0)
    failure = failure_count_dist.get(count, 0)
    total = success + failure
    rate = success / total * 100 if total > 0 else 0
    if rate < 50:
        print(f"  • {count} 隻共識馬: {rate:.1f}% 成功")

for race in sorted(success_race_dist.keys(), key=lambda x: int(x.replace('R',''))):
    count = success_race_dist[race]
    total = count + failure_race_dist.get(race, 0)
    rate = count / total * 100 if total > 0 else 0
    if rate < 50:
        print(f"  • {race} 場: {rate:.1f}% 成功")
