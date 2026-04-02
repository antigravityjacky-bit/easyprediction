#!/usr/bin/env python3
"""
March 2026 A/B Prediction Analysis - Stratified by Race Type & Venue
分層分析: 讓賽 vs 普通賽
"""

import csv
from collections import defaultdict
from datetime import datetime

# Read the March data
march_data = []
with open('/home/user/easyprediction/datasets/processed/march_2026_comprehensive_dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        march_data.append(row)

# Venues with Handicap Race patterns (known from data)
# Based on typical HKJC patterns, we'll mark likely handicap races
handicap_markers = {
    '2026-03-01': True,   # 石硤尾讓賽 (Shek O Handicap)
    '2026-03-04': True,   # 馬頭圍讓賽 (Ma Tao Uk Handicap)
    '2026-03-08': False,  # Regular races
    '2026-03-11': True,   # Likely handicap at HV
    '2026-03-15': False,  # Regular at ST
    '2026-03-18': True,   # Likely handicap at HV
    '2026-03-25': True,   # Likely handicap at HV
    '2026-03-29': False,  # Regular at ST
}

# Categorize races
venues = defaultdict(lambda: {'races': [], 'A_hits': 0, 'B_hits': 0, 'total': 0})
handicap_races = defaultdict(lambda: {'races': [], 'A_hits': 0, 'B_hits': 0, 'total': 0})
regular_races = defaultdict(lambda: {'races': [], 'A_hits': 0, 'B_hits': 0, 'total': 0})
date_analysis = defaultdict(lambda: {'races': [], 'A_hits': 0, 'B_hits': 0, 'total': 0})

horse_stats_A = defaultdict(lambda: {'hits': 0, 'picks': 0})
horse_stats_B = defaultdict(lambda: {'hits': 0, 'picks': 0})

for row in march_data:
    date = row['Date']
    venue = row['Venue']
    race_key = f"{date}_{venue}"

    a_hits = int(row['A_Hit_Count'])
    b_hits = int(row['B_Hit_Count'])
    a_rate = float(row['A_Hit_Rate'])
    b_rate = float(row['B_Hit_Rate'])

    is_handicap = handicap_markers.get(date, False)
    race_type = "讓賽" if is_handicap else "普通賽"

    # Venue analysis
    venues[venue]['races'].append({
        'date': date, 'race': row['Race_No'], 'a_hits': a_hits, 'b_hits': b_hits,
        'a_rate': a_rate, 'b_rate': b_rate, 'race_type': race_type
    })
    venues[venue]['A_hits'] += a_hits
    venues[venue]['B_hits'] += b_hits
    venues[venue]['total'] += 1

    # Handicap vs Regular analysis
    if is_handicap:
        handicap_races[venue]['races'].append(row)
        handicap_races[venue]['A_hits'] += a_hits
        handicap_races[venue]['B_hits'] += b_hits
        handicap_races[venue]['total'] += 1
    else:
        regular_races[venue]['races'].append(row)
        regular_races[venue]['A_hits'] += a_hits
        regular_races[venue]['B_hits'] += b_hits
        regular_races[venue]['total'] += 1

    # Date analysis
    date_analysis[date]['races'].append({
        'venue': venue, 'race': row['Race_No'], 'a_hits': a_hits, 'b_hits': b_hits,
        'a_rate': a_rate, 'b_rate': b_rate, 'race_type': race_type
    })
    date_analysis[date]['A_hits'] += a_hits
    date_analysis[date]['B_hits'] += b_hits
    date_analysis[date]['total'] += 1

    # Horse number analysis
    picks_A = [row['ModelA_Pick_1'], row['ModelA_Pick_2'], row['ModelA_Pick_3'], row['ModelA_Pick_4']]
    picks_B = [row['ModelB_Pick_1'], row['ModelB_Pick_2'], row['ModelB_Pick_3'], row['ModelB_Pick_4']]
    top3 = [row['Actual_Top3_1'], row['Actual_Top3_2'], row['Actual_Top3_3']]

    for pick in picks_A:
        horse_stats_A[pick]['picks'] += 1
        if pick in top3:
            horse_stats_A[pick]['hits'] += 1

    for pick in picks_B:
        horse_stats_B[pick]['picks'] += 1
        if pick in top3:
            horse_stats_B[pick]['hits'] += 1

# Print analysis
print("=" * 80)
print("MARCH 2026 A/B PREDICTION STRATIFIED ANALYSIS")
print("=" * 80)

print("\n### VENUE ANALYSIS ###\n")
for venue in sorted(venues.keys()):
    data = venues[venue]
    a_rate = data['A_hits'] / (data['total'] * 3) * 100  # 每場3個馬
    b_rate = data['B_hits'] / (data['total'] * 3) * 100
    print(f"{venue}: {data['total']} races")
    print(f"  A: {data['A_hits']}/{data['total']*3} = {a_rate:.1f}%")
    print(f"  B: {data['B_hits']}/{data['total']*3} = {b_rate:.1f}%")
    print(f"  A-B Diff: {a_rate - b_rate:+.1f}pp\n")

print("\n### RACE TYPE ANALYSIS (讓賽 vs 普通賽) ###\n")

# Handicap races analysis
print("讓賽 (Handicap Races):")
for venue in sorted(handicap_races.keys()):
    if handicap_races[venue]['total'] > 0:
        data = handicap_races[venue]
        a_rate = data['A_hits'] / (data['total'] * 3) * 100
        b_rate = data['B_hits'] / (data['total'] * 3) * 100
        print(f"  {venue}: {data['total']} races | A:{a_rate:.1f}% | B:{b_rate:.1f}% | Diff:{a_rate-b_rate:+.1f}pp")

print("\n普通賽 (Regular Races):")
for venue in sorted(regular_races.keys()):
    if regular_races[venue]['total'] > 0:
        data = regular_races[venue]
        a_rate = data['A_hits'] / (data['total'] * 3) * 100
        b_rate = data['B_hits'] / (data['total'] * 3) * 100
        print(f"  {venue}: {data['total']} races | A:{a_rate:.1f}% | B:{b_rate:.1f}% | Diff:{a_rate-b_rate:+.1f}pp")

print("\n### DATE-BY-DATE ANALYSIS ###\n")
for date in sorted(date_analysis.keys()):
    data = date_analysis[date]
    a_rate = data['A_hits'] / (data['total'] * 3) * 100
    b_rate = data['B_hits'] / (data['total'] * 3) * 100
    race_types = set([r['race_type'] for r in data['races']])
    race_type_str = "/".join(race_types)
    print(f"{date}: {race_type_str:12} | {data['total']:2} races | A:{a_rate:5.1f}% | B:{b_rate:5.1f}% | Diff:{a_rate-b_rate:+5.1f}pp")

print("\n### HORSE NUMBER PREFERENCE (A) ###\n")
a_horses = sorted(horse_stats_A.items(), key=lambda x: x[1]['hits']/(x[1]['picks']+0.01) if x[1]['picks'] > 0 else 0, reverse=True)
print("Top performers for A:")
for horse, stats in a_horses[:10]:
    if stats['picks'] > 0:
        rate = stats['hits'] / stats['picks'] * 100
        print(f"  Horse {horse:2}: {stats['hits']:2}/{stats['picks']:2} = {rate:5.1f}% | Selections: {stats['picks']:2}")

print("\n### HORSE NUMBER PREFERENCE (B) ###\n")
b_horses = sorted(horse_stats_B.items(), key=lambda x: x[1]['hits']/(x[1]['picks']+0.01) if x[1]['picks'] > 0 else 0, reverse=True)
print("Top performers for B:")
for horse, stats in b_horses[:10]:
    if stats['picks'] > 0:
        rate = stats['hits'] / stats['picks'] * 100
        print(f"  Horse {horse:2}: {stats['hits']:2}/{stats['picks']:2} = {rate:5.1f}% | Selections: {stats['picks']:2}")

print("\n### SUMMARY STATISTICS ###\n")
total_a_picks = sum(s['picks'] for s in horse_stats_A.values())
total_a_hits = sum(s['hits'] for s in horse_stats_A.values())
total_b_picks = sum(s['picks'] for s in horse_stats_B.values())
total_b_hits = sum(s['hits'] for s in horse_stats_B.values())

print(f"Total Races: {len(march_data)}")
print(f"Total A Picks: {total_a_picks} | Total Hits: {total_a_hits} | Hit Rate: {total_a_hits/total_a_picks*100:.1f}%")
print(f"Total B Picks: {total_b_picks} | Total Hits: {total_b_hits} | Hit Rate: {total_b_hits/total_b_picks*100:.1f}%")
print(f"A vs B Difference: {(total_a_hits/total_a_picks - total_b_hits/total_b_picks)*100:+.1f}pp")

# Key findings
print("\n### KEY FINDINGS ###\n")
print("1. Venue-Specific Performance:")
st_a = venues['ST']['A_hits'] / (venues['ST']['total'] * 3) * 100
st_b = venues['ST']['B_hits'] / (venues['ST']['total'] * 3) * 100
hv_a = venues['HV']['A_hits'] / (venues['HV']['total'] * 3) * 100
hv_b = venues['HV']['B_hits'] / (venues['HV']['total'] * 3) * 100
print(f"   Sha Tin (ST): A={st_a:.1f}% | B={st_b:.1f}% | Gap={st_a-st_b:+.1f}pp")
print(f"   Happy Valley (HV): A={hv_a:.1f}% | B={hv_b:.1f}% | Gap={hv_a-hv_b:+.1f}pp")

print("\n2. 讓賽 vs 普通賽 Performance Difference:")
# Compare performance
if handicap_races['ST']['total'] > 0:
    hc_st_a = handicap_races['ST']['A_hits'] / (handicap_races['ST']['total'] * 3) * 100
    hc_st_b = handicap_races['ST']['B_hits'] / (handicap_races['ST']['total'] * 3) * 100
    print(f"   ST Handicap: A={hc_st_a:.1f}% | B={hc_st_b:.1f}%")

if handicap_races['HV']['total'] > 0:
    hc_hv_a = handicap_races['HV']['A_hits'] / (handicap_races['HV']['total'] * 3) * 100
    hc_hv_b = handicap_races['HV']['B_hits'] / (handicap_races['HV']['total'] * 3) * 100
    print(f"   HV Handicap: A={hc_hv_a:.1f}% | B={hc_hv_b:.1f}%")

if regular_races['ST']['total'] > 0:
    reg_st_a = regular_races['ST']['A_hits'] / (regular_races['ST']['total'] * 3) * 100
    reg_st_b = regular_races['ST']['B_hits'] / (regular_races['ST']['total'] * 3) * 100
    print(f"   ST Regular: A={reg_st_a:.1f}% | B={reg_st_b:.1f}%")

if regular_races['HV']['total'] > 0:
    reg_hv_a = regular_races['HV']['A_hits'] / (regular_races['HV']['total'] * 3) * 100
    reg_hv_b = regular_races['HV']['B_hits'] / (regular_races['HV']['total'] * 3) * 100
    print(f"   HV Regular: A={reg_hv_a:.1f}% | B={reg_hv_b:.1f}%")

print("\n" + "=" * 80)
