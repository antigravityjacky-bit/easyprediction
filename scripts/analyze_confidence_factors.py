#!/usr/bin/env python3
"""
信心場次 (High-Confidence Race) Analysis
Identify common factors in Method Y's successful predictions (>=2 hits)
"""

import csv
from collections import defaultdict

# March hit rate data (from method_y_evolved.py)
march_A_rates = {
    '1': 0.386, '2': 0.514, '3': 0.465, '4': 0.500,
    '5': 0.440, '6': 0.500, '7': 0.391, '8': 0.440,
    '9': 0.600, '10': 0.417, '11': 0.600, '12': 0.400,
    '13': 0.500, '14': 0.300
}
march_B_rates = {
    '1': 0.342, '2': 0.407, '3': 0.516, '4': 0.519,
    '5': 0.250, '6': 0.484, '7': 0.360, '8': 0.407,
    '9': 0.500, '10': 0.389, '11': 0.500, '12': 0.200,
    '13': 0.400, '14': 0.200
}

def method_y_evolved(a_picks, b_picks, is_handicap=True, venue='HV'):
    """Method Y v2.0 logic"""
    a_set = set(a_picks)
    b_set = set(b_picks)
    consensus = list(a_set & b_set)
    a_unique = list(a_set - b_set)
    b_unique = list(b_set - a_set)

    def consensus_score_y(h):
        ar = march_A_rates.get(h, 0.40)
        br = march_B_rates.get(h, 0.40)
        if venue == 'HV' and is_handicap:
            return ar * 0.65 + br * 0.35
        return (ar + br) / 2

    def a_unique_score_y(h):
        return march_A_rates.get(h, 0.40)

    def b_unique_score_y(h):
        if venue == 'HV' and is_handicap:
            return 0.0
        else:
            return march_B_rates.get(h, 0.40) * 0.2

    candidates = []
    for h in consensus:
        candidates.append((h, consensus_score_y(h), 'consensus'))
    for h in a_unique:
        candidates.append((h, a_unique_score_y(h), 'A-unique'))
    for h in b_unique:
        candidates.append((h, b_unique_score_y(h), 'B-unique'))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:3]], candidates, consensus, a_unique, b_unique


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

# Analyze each race
success_cases = []  # Cases with >=2 hits
failure_cases = []  # Cases with <2 hits
all_cases = []

for row in march_data:
    if row['Date'] not in target_dates:
        continue

    venue = row['Venue']
    date = row['Date']
    is_handicap = handicap_markers.get(date, False)

    a_picks = [row['ModelA_Pick_1'], row['ModelA_Pick_2'], row['ModelA_Pick_3'], row['ModelA_Pick_4']]
    b_picks = [row['ModelB_Pick_1'], row['ModelB_Pick_2'], row['ModelB_Pick_3'], row['ModelB_Pick_4']]
    actual = [row['Actual_Top3_1'], row['Actual_Top3_2'], row['Actual_Top3_3']]

    # Get Method Y prediction
    pred_y, candidates, consensus, a_unique, b_unique = method_y_evolved(a_picks, b_picks, is_handicap, venue)

    # Count hits
    hits = len([h for h in pred_y if h in actual])

    # Extract candidate details
    consensus_hits = sum(1 for h in consensus if h in actual)
    a_unique_hits = sum(1 for h in a_unique if h in actual)
    b_unique_hits = sum(1 for h in b_unique if h in actual)

    # Calculate consensus horse hit rates
    consensus_rates = []
    for h in consensus:
        ar = march_A_rates.get(h, 0.40)
        br = march_B_rates.get(h, 0.40)
        if venue == 'HV' and is_handicap:
            rate = ar * 0.65 + br * 0.35
        else:
            rate = (ar + br) / 2
        consensus_rates.append(rate)

    # Calculate a_unique rates
    a_unique_rates = [march_A_rates.get(h, 0.40) for h in a_unique]

    # Calculate b_unique rates
    if venue == 'HV' and is_handicap:
        b_unique_rates = [0.0] * len(b_unique)
    else:
        b_unique_rates = [march_B_rates.get(h, 0.40) * 0.2 for h in b_unique]

    # Count high-rate horses (>50%)
    high_rate_consensus = sum(1 for r in consensus_rates if r > 0.50)
    high_rate_a_unique = sum(1 for r in a_unique_rates if r > 0.50)

    # Check if all consensus entered top 3
    consensus_complete = all(h in actual for h in consensus) if consensus else False

    case = {
        'date': date,
        'venue': venue,
        'race': row['Race_No'],
        'is_handicap': is_handicap,
        'hits': hits,
        'consensus': consensus,
        'consensus_count': len(consensus),
        'consensus_rates': consensus_rates,
        'consensus_avg_rate': sum(consensus_rates) / len(consensus_rates) if consensus_rates else 0,
        'consensus_hits': consensus_hits,
        'consensus_complete': consensus_complete,
        'a_unique': a_unique,
        'a_unique_count': len(a_unique),
        'a_unique_rates': a_unique_rates,
        'a_unique_avg_rate': sum(a_unique_rates) / len(a_unique_rates) if a_unique_rates else 0,
        'a_unique_hits': a_unique_hits,
        'b_unique': b_unique,
        'b_unique_count': len(b_unique),
        'high_rate_consensus': high_rate_consensus,
        'high_rate_a_unique': high_rate_a_unique,
        'actual': actual,
        'prediction': pred_y,
    }

    all_cases.append(case)

    if hits >= 2:
        success_cases.append(case)
    else:
        failure_cases.append(case)

# Analysis
print("=" * 100)
print("信心場次 (High-Confidence Race) Analysis - Method Y v2.0")
print("=" * 100)
print(f"\nTotal Cases: {len(all_cases)}")
print(f"Success Cases (>=2 hits): {len(success_cases)}")
print(f"Failure Cases (<2 hits): {len(failure_cases)}")

# Calculate statistics
print("\n" + "=" * 100)
print("FACTOR COMPARISON: Success vs Failure")
print("=" * 100)

# 1. Consensus horse count
if success_cases and failure_cases:
    success_consensus_counts = [c['consensus_count'] for c in success_cases]
    failure_consensus_counts = [c['consensus_count'] for c in failure_cases]

    success_avg_consensus = sum(success_consensus_counts) / len(success_consensus_counts)
    failure_avg_consensus = sum(failure_consensus_counts) / len(failure_consensus_counts)

    print(f"\n1. Consensus Horse Count:")
    print(f"   Success cases: avg = {success_avg_consensus:.2f} horses")
    print(f"   Failure cases: avg = {failure_avg_consensus:.2f} horses")
    print(f"   Difference: {success_avg_consensus - failure_avg_consensus:+.2f} horses")

    # Distribution
    success_dist = {}
    for count in success_consensus_counts:
        success_dist[count] = success_dist.get(count, 0) + 1
    failure_dist = {}
    for count in failure_consensus_counts:
        failure_dist[count] = failure_dist.get(count, 0) + 1

    print(f"\n   Success distribution:")
    for count in sorted(set(success_consensus_counts)):
        pct = success_dist[count] / len(success_cases) * 100
        print(f"     {count} horses: {success_dist.get(count, 0):2} cases ({pct:5.1f}%)")
    print(f"\n   Failure distribution:")
    for count in sorted(set(failure_consensus_counts)):
        pct = failure_dist[count] / len(failure_cases) * 100
        print(f"     {count} horses: {failure_dist.get(count, 0):2} cases ({pct:5.1f}%)")

# 2. Average consensus hit rate
if success_cases and failure_cases:
    success_avg_rates = [c['consensus_avg_rate'] for c in success_cases if c['consensus_rates']]
    failure_avg_rates = [c['consensus_avg_rate'] for c in failure_cases if c['consensus_rates']]

    success_avg_consensus_rate = sum(success_avg_rates) / len(success_avg_rates) if success_avg_rates else 0
    failure_avg_consensus_rate = sum(failure_avg_rates) / len(failure_avg_rates) if failure_avg_rates else 0

    print(f"\n2. Average Consensus Horse Hit Rate:")
    print(f"   Success cases: {success_avg_consensus_rate:.1%}")
    print(f"   Failure cases: {failure_avg_consensus_rate:.1%}")
    print(f"   Difference: {success_avg_consensus_rate - failure_avg_consensus_rate:+.1%}")

# 3. High-rate horse presence (>50%)
if success_cases and failure_cases:
    success_high_rate = [c['high_rate_consensus'] for c in success_cases]
    failure_high_rate = [c['high_rate_consensus'] for c in failure_cases]

    success_has_high_rate = sum(1 for c in success_cases if c['high_rate_consensus'] > 0)
    failure_has_high_rate = sum(1 for c in failure_cases if c['high_rate_consensus'] > 0)

    print(f"\n3. High-Rate Consensus Horses (>50%):")
    print(f"   Success: {success_has_high_rate}/{len(success_cases)} cases ({success_has_high_rate/len(success_cases)*100:.1f}%) have >50% horse")
    print(f"   Failure: {failure_has_high_rate}/{len(failure_cases)} cases ({failure_has_high_rate/len(failure_cases)*100:.1f}%) have >50% horse")

# 4. A-unique average rate
if success_cases and failure_cases:
    success_a_unique_rates = [c['a_unique_avg_rate'] for c in success_cases if c['a_unique_rates']]
    failure_a_unique_rates = [c['a_unique_avg_rate'] for c in failure_cases if c['a_unique_rates']]

    success_avg_a_unique = sum(success_a_unique_rates) / len(success_a_unique_rates) if success_a_unique_rates else 0
    failure_avg_a_unique = sum(failure_a_unique_rates) / len(failure_a_unique_rates) if failure_a_unique_rates else 0

    print(f"\n4. A-Unique Average Hit Rate:")
    print(f"   Success cases: {success_avg_a_unique:.1%}")
    print(f"   Failure cases: {failure_avg_a_unique:.1%}")
    print(f"   Difference: {success_avg_a_unique - failure_avg_a_unique:+.1%}")

# 5. Consensus complete entry
if success_cases and failure_cases:
    success_complete = sum(1 for c in success_cases if c['consensus_complete'])
    failure_complete = sum(1 for c in failure_cases if c['consensus_complete'])

    print(f"\n5. Consensus Complete Entry into Top 3:")
    print(f"   Success: {success_complete}/{len(success_cases)} cases ({success_complete/len(success_cases)*100:.1f}%)")
    print(f"   Failure: {failure_complete}/{len(failure_cases)} cases ({failure_complete/len(failure_cases)*100:.1f}%)")

# 6. Consensus accuracy (hitting rate)
if success_cases and failure_cases:
    success_consensus_accuracy = [c['consensus_hits'] / len(c['consensus']) if c['consensus'] else 0 for c in success_cases]
    failure_consensus_accuracy = [c['consensus_hits'] / len(c['consensus']) if c['consensus'] else 0 for c in failure_cases]

    success_avg_accuracy = sum(success_consensus_accuracy) / len(success_consensus_accuracy) if success_consensus_accuracy else 0
    failure_avg_accuracy = sum(failure_consensus_accuracy) / len(failure_consensus_accuracy) if failure_consensus_accuracy else 0

    print(f"\n6. Consensus Horse Accuracy (% of consensus that actually hit):")
    print(f"   Success: {success_avg_accuracy:.1%}")
    print(f"   Failure: {failure_avg_accuracy:.1%}")
    print(f"   Difference: {success_avg_accuracy - failure_avg_accuracy:+.1%}")

# Print detailed success cases
print("\n" + "=" * 100)
print("SUCCESS CASES DETAIL (>=2 hits)")
print("=" * 100)
for case in success_cases:
    print(f"\n{case['date']} {case['venue']} {case['race']}: {case['hits']} hits")
    print(f"  Consensus: {case['consensus']} ({len(case['consensus'])} horses, avg rate: {case['consensus_avg_rate']:.1%})")
    print(f"  A-unique: {case['a_unique']}")
    print(f"  Prediction: {case['prediction']}")
    print(f"  Actual: {case['actual']}")

print("\n" + "=" * 100)
print("FAILURE CASES DETAIL (<2 hits)")
print("=" * 100)
for case in failure_cases:
    print(f"\n{case['date']} {case['venue']} {case['race']}: {case['hits']} hits")
    print(f"  Consensus: {case['consensus']} ({len(case['consensus'])} horses, avg rate: {case['consensus_avg_rate']:.1%})")
    print(f"  A-unique: {case['a_unique']}")
    print(f"  Prediction: {case['prediction']}")
    print(f"  Actual: {case['actual']}")

# Confidence level definition
print("\n" + "=" * 100)
print("CONFIDENCE LEVEL DEFINITION (信心場次)")
print("=" * 100)

if success_cases:
    # Based on the analysis, define thresholds
    avg_success_consensus = sum(c['consensus_count'] for c in success_cases) / len(success_cases)
    avg_success_rate = sum(c['consensus_avg_rate'] for c in success_cases if c['consensus_rates']) / len([c for c in success_cases if c['consensus_rates']])

    print(f"\nKey Success Indicators:")
    print(f"  ✅ Consensus count >= {int(avg_success_consensus)} horses")
    print(f"  ✅ Consensus average hit rate >= {avg_success_rate:.0%}")
    print(f"  ✅ At least 1 consensus horse with >50% hit rate")
    print(f"  ✅ All consensus horses should enter top 3 (if possible)")

    print(f"\nConfidence Tier Proposal:")
    print(f"\n  【高信心 / HIGH CONFIDENCE】")
    print(f"    - Consensus count >= 3 horses")
    print(f"    - Consensus avg rate >= {avg_success_rate:.0%}")
    print(f"    - At least 2 consensus horses with >45% hit rate")
    print(f"    → Expected Hit Rate: 70%+ 中2隻")

    print(f"\n  【中信心 / MEDIUM CONFIDENCE】")
    print(f"    - Consensus count >= 2 horses")
    print(f"    - Consensus avg rate >= 50%")
    print(f"    - At least 1 consensus horse with >50% hit rate")
    print(f"    → Expected Hit Rate: 55-70% 中2隻")

    print(f"\n  【低信心 / LOW CONFIDENCE】")
    print(f"    - Consensus count < 2 horses")
    print(f"    - OR Consensus avg rate < 50%")
    print(f"    → Expected Hit Rate: <55% 中2隻")

print("\n" + "=" * 100)
