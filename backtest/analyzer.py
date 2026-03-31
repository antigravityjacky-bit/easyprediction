"""
Backtest Results Analyzer

Generates backtest_summary.md with:
  - Overall metrics (precision@3, recall@3, hit rate)
  - Breakdown by venue, distance, condition
  - Counter-trend analysis
  - Failure case analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


# ── Distance buckets ─────────────────────────────────────────────────────────

def _distance_bucket(dist: int) -> str:
    if dist <= 1400:
        return "短途 (1000-1400m)"
    elif dist <= 1650:
        return "一哩 (1600-1650m)"
    elif dist <= 2000:
        return "中距 (1800-2000m)"
    else:
        return "長途 (2200m+)"


# ── Main analyzer ────────────────────────────────────────────────────────────

def generate_backtest_summary(
    results_path: Path | None = None,
    predictions_path: Path | None = None,
    output_dir: Path | None = None,
) -> str:
    """
    Generate backtest_summary.md from backtest results.

    Returns the markdown content as a string.
    """
    default_dir = Path(__file__).resolve().parents[3] / "datasets" / "processed" / "backtest"
    output_dir = output_dir or default_dir

    if results_path is None:
        results_path = output_dir / "backtest_results.csv"
    if predictions_path is None:
        predictions_path = output_dir / "prediction_report.csv"

    if not results_path.exists():
        return "# Backtest Summary\n\nNo results found. Run the backtest first."

    df = pd.read_csv(results_path)
    pred_df = pd.read_csv(predictions_path) if predictions_path.exists() else None

    lines = []
    lines.append("# 賽馬預測回測報告 (Backtest Summary)")
    lines.append("")
    lines.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ── Overall Metrics ──────────────────────────────────────────────────
    lines.append("## 1. 整體指標")
    lines.append("")

    n_races = len(df)
    avg_precision = df["precision_at_3"].mean()
    avg_recall = df["recall_at_3"].mean() if "recall_at_3" in df.columns else avg_precision
    hit_rate = df["hit_any"].mean()
    exact_3 = (df["correct_count"] == 3).mean()
    exact_2 = (df["correct_count"] >= 2).mean()
    exact_1 = (df["correct_count"] >= 1).mean()
    avg_correct = df["correct_count"].mean()

    lines.append(f"| 指標 | 數值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 總場次 | {n_races} |")
    lines.append(f"| 平均 Precision@3 | {avg_precision:.3f} ({avg_precision*100:.1f}%) |")
    lines.append(f"| 平均 Recall@3 | {avg_recall:.3f} ({avg_recall*100:.1f}%) |")
    lines.append(f"| 命中率（至少1匹正確） | {hit_rate:.3f} ({hit_rate*100:.1f}%) |")
    lines.append(f"| 命中≥2匹正確 | {exact_2:.3f} ({exact_2*100:.1f}%) |")
    lines.append(f"| 完美命中（3匹全中） | {exact_3:.3f} ({exact_3*100:.1f}%) |")
    lines.append(f"| 平均正確匹數 | {avg_correct:.2f}/3 |")
    lines.append("")

    # Random baseline
    # For a 14-horse field, P(at least 1 correct in top-3 pick) ≈ 1 - C(11,3)/C(14,3)
    lines.append(f"> 隨機基準 (14馬場): Precision@3 ≈ 21.4%, 命中率 ≈ 54.4%")
    lines.append("")

    # ── Venue Breakdown ──────────────────────────────────────────────────
    lines.append("## 2. 場地分析")
    lines.append("")
    lines.append("| 場地 | 場次 | Precision@3 | 命中率 | 平均正確 |")
    lines.append("|------|------|-------------|--------|----------|")

    for venue in ["ST", "HV"]:
        vdf = df[df["venue_code"] == venue]
        if len(vdf) == 0:
            continue
        venue_name = "沙田" if venue == "ST" else "跑馬地"
        lines.append(
            f"| {venue_name} ({venue}) | {len(vdf)} | "
            f"{vdf['precision_at_3'].mean():.3f} | "
            f"{vdf['hit_any'].mean():.3f} | "
            f"{vdf['correct_count'].mean():.2f} |"
        )
    lines.append("")

    # ── Distance Breakdown ───────────────────────────────────────────────
    lines.append("## 3. 距離分析")
    lines.append("")

    df["distance_bucket"] = df["distance"].apply(_distance_bucket)
    lines.append("| 距離 | 場次 | Precision@3 | 命中率 | 平均正確 |")
    lines.append("|------|------|-------------|--------|----------|")

    for bucket in sorted(df["distance_bucket"].unique()):
        bdf = df[df["distance_bucket"] == bucket]
        lines.append(
            f"| {bucket} | {len(bdf)} | "
            f"{bdf['precision_at_3'].mean():.3f} | "
            f"{bdf['hit_any'].mean():.3f} | "
            f"{bdf['correct_count'].mean():.2f} |"
        )
    lines.append("")

    # ── Condition Breakdown ──────────────────────────────────────────────
    lines.append("## 4. 場地狀況分析")
    lines.append("")
    lines.append("| 場地狀況 | 場次 | Precision@3 | 命中率 |")
    lines.append("|----------|------|-------------|--------|")

    for cond in sorted(df["condition"].unique()):
        cdf = df[df["condition"] == cond]
        lines.append(
            f"| {cond} | {len(cdf)} | "
            f"{cdf['precision_at_3'].mean():.3f} | "
            f"{cdf['hit_any'].mean():.3f} |"
        )
    lines.append("")

    # ── Field Size Analysis ──────────────────────────────────────────────
    lines.append("## 5. 出馬數分析")
    lines.append("")

    df["field_bucket"] = pd.cut(
        df["field_size"],
        bins=[0, 9, 12, 20],
        labels=["小場 (≤9)", "中場 (10-12)", "大場 (>12)"],
    )
    lines.append("| 出馬數 | 場次 | Precision@3 | 命中率 |")
    lines.append("|--------|------|-------------|--------|")

    for bucket in df["field_bucket"].cat.categories:
        fdf = df[df["field_bucket"] == bucket]
        if len(fdf) == 0:
            continue
        lines.append(
            f"| {bucket} | {len(fdf)} | "
            f"{fdf['precision_at_3'].mean():.3f} | "
            f"{fdf['hit_any'].mean():.3f} |"
        )
    lines.append("")

    # ── Counter-Trend Analysis ───────────────────────────────────────────
    lines.append("## 6. 反趨勢馬匹分析")
    lines.append("")

    ct_placed = 0
    ct_total_flagged = 0
    for _, row in df.iterrows():
        try:
            ct_horses = json.loads(row.get("ct_horses_placed", "[]"))
            ct_placed += len(ct_horses)
        except (json.JSONDecodeError, TypeError):
            pass

    if pred_df is not None and "ct_composite" in pred_df.columns:
        ct_flagged = pred_df[pred_df["ct_composite"] >= 40]
        ct_total_flagged = len(ct_flagged)
        ct_in_top3 = ct_flagged[ct_flagged["actual_position"].fillna(99) <= 3]
        ct_hit_rate = len(ct_in_top3) / ct_total_flagged if ct_total_flagged > 0 else 0

        lines.append(f"| 指標 | 數值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 反趨勢標記馬匹總數 (composite ≥ 40) | {ct_total_flagged} |")
        lines.append(f"| 其中跑入前3名 | {len(ct_in_top3)} |")
        lines.append(f"| 反趨勢命中率 | {ct_hit_rate:.3f} ({ct_hit_rate*100:.1f}%) |")
    else:
        lines.append("反趨勢數據不可用。")
    lines.append("")

    # ── Failure Case Analysis ────────────────────────────────────────────
    lines.append("## 7. 失敗案例分析")
    lines.append("")

    failures = df[df["correct_count"] == 0]
    lines.append(f"完全失敗場次（0/3 正確）: {len(failures)}/{n_races} ({len(failures)/n_races*100:.1f}%)")
    lines.append("")

    if len(failures) > 0:
        # Breakdown failures by venue
        lines.append("### 失敗場次特徵:")
        lines.append("")
        fail_venues = failures["venue_code"].value_counts()
        for v, c in fail_venues.items():
            lines.append(f"- {v}: {c}場")

        fail_distances = failures["distance_bucket"].value_counts()
        for d, c in fail_distances.items():
            lines.append(f"- {d}: {c}場")

        avg_field_fail = failures["field_size"].mean()
        avg_field_all = df["field_size"].mean()
        lines.append(f"- 失敗場次平均出馬數: {avg_field_fail:.1f} (整體: {avg_field_all:.1f})")
        lines.append("")

    # ── ML Model Status ──────────────────────────────────────────────────
    lines.append("## 8. 模型狀態")
    lines.append("")
    has_ml = df["has_ml_models"].any() if "has_ml_models" in df.columns else False
    if has_ml:
        lines.append("✓ ML 模型已載入（LightGBM + XGBoost + LambdaRank）")
    else:
        lines.append("⚠ 僅使用物理模型（未載入 ML 模型）")
        lines.append("  - 使用 `python -m horseracing.ml.train` 訓練模型後重新回測")
    lines.append("")

    # ── Conclusion ───────────────────────────────────────────────────────
    lines.append("## 9. 結論與建議")
    lines.append("")
    if avg_precision > 0.3:
        lines.append("模型表現良好，Precision@3 超過 30%（隨機基準 ~21%）。")
    elif avg_precision > 0.21:
        lines.append("模型表現略優於隨機基準，建議進一步優化特徵或增加訓練數據。")
    else:
        lines.append("模型表現接近或低於隨機基準，需要重新檢視特徵工程和模型選擇。")

    lines.append("")
    lines.append("### 改善方向:")
    lines.append("1. 收集更多歷史數據（增加訓練集大小）")
    lines.append("2. 增加馬匹個人頁面的詳細分段時間")
    lines.append("3. 加入騎師/練馬師表現特徵")
    lines.append("4. 針對跑馬地的彎道物理模型進一步校準")
    lines.append("")

    # Write file
    content = "\n".join(lines)
    summary_path = output_dir / "backtest_summary.md"
    summary_path.write_text(content, encoding="utf-8")
    print(f"Summary written to: {summary_path}")

    return content


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("--results", default=None, help="Path to backtest_results.csv")
    parser.add_argument("--predictions", default=None, help="Path to prediction_report.csv")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    generate_backtest_summary(
        results_path=Path(args.results) if args.results else None,
        predictions_path=Path(args.predictions) if args.predictions else None,
        output_dir=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
