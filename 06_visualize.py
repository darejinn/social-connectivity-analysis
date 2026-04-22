"""
06_visualize.py
===============
발표 그림 11~16을 재생성.

각 함수는 독립적으로 호출 가능하며, 분석 결과 DataFrame을 받아 .png 파일을 저장.

재현되는 그림:
    그림 11  — 3축 레이더 (관계별)           : plot_three_axis_radar()
    그림 12  — 대화 Level 분포 바              : plot_level_distribution()
    그림 13  — 시간대 × 요일 히트맵            : plot_temporal_heatmap()
    그림 14  — 응답 시간 분포 (log-scale hist) : plot_response_time()
    그림 15  — 종합 대시보드 (6-panel)         : plot_dashboard()
    그림 16  — Fermi 건강 영향 (HR + 수명)     : plot_health_impact()

수정 포인트:
    - COLORS: 관계별 색상
    - 각 plot_ 함수의 figsize, 타이틀, 축 설정
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정 (macOS/Linux 공통 대응)
plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 관계별 색상 (발표 그림과 일치)
COLORS = {
    "가족톡": "#E63946",  # 빨강
    "단톡A":  "#4A7A96",  # 파랑
    "단톡B":  "#2A9D8F",  # 초록
}
AXIS_COLORS = {"Structural": "#4A7A96", "Functional": "#E63946", "Quality": "#F4A261"}


# ================================================================
# 그림 11 — 3축 레이더
# ================================================================
def plot_three_axis_radar(scores: pd.DataFrame, output: Path) -> None:
    """
    scores: 04 단계 산출물 (관계별 × Structural/Functional/Quality)

    관계별 레이더를 가로 나열. "Quality 축 함몰"을 시각화.
    """
    relationships = [r for r in scores.index if r != "_OVERALL_"]
    axes_labels = ["Structural\n(Quantity)", "Functional\n(Perceived)", "Quality\n(Depth)"]
    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # 닫힘

    fig, axs = plt.subplots(
        1, len(relationships),
        figsize=(5 * len(relationships), 5),
        subplot_kw={"projection": "polar"},
    )
    if len(relationships) == 1:
        axs = [axs]

    for ax, rel in zip(axs, relationships):
        values = [scores.loc[rel, "Structural"], scores.loc[rel, "Functional"], scores.loc[rel, "Quality"]]
        values += values[:1]

        color = COLORS.get(rel, "gray")
        ax.plot(angles, values, linewidth=2.5, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes_labels, fontsize=11)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9)
        ax.set_title(f"{rel}\n(SCI {scores.loc[rel, 'SCI']:.1f}/10)", fontsize=13, pad=15)

    fig.suptitle("Holt-Lunstad 3-Axis Profile by Relationship", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [그림 11] {output}")


# ================================================================
# 그림 12 — 대화 Level 분포
# ================================================================
def plot_level_distribution(level_pct: pd.DataFrame, output: Path, mehl_threshold: float = 30.0) -> None:
    """
    level_pct: 02 단계의 level_distribution() 결과 (행=관계, 열=L1~L5)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    levels = ["L1", "L2", "L3", "L4", "L5"]
    level_labels = ["L1\nTransactional", "L2\nSmall talk", "L3\nInformational", "L4\nOpinion", "L5\nEmotional"]
    x = np.arange(len(levels))
    width = 0.25

    rels = list(level_pct.index)
    for i, rel in enumerate(rels):
        vals = [level_pct.loc[rel, lv] for lv in levels]
        offset = (i - (len(rels) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=rel, color=COLORS.get(rel, "gray"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%",
                    ha="center", fontsize=8)

    # Substantive conversation 영역 음영
    ax.axvspan(2.5, 4.5, alpha=0.15, color="#2E7D32",
               label=f"Mehl 2010 substantive (≈{mehl_threshold:.0f}% healthy)")

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Conversation Level Distribution Across Relationships\n(Mehl 2010 \"substantive conversation\" framework)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(60, level_pct.values.max() + 10))

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [그림 12] {output}")


# ================================================================
# 그림 13 — 시간대 × 요일 히트맵
# ================================================================
def plot_temporal_heatmap(heatmap_data: pd.DataFrame, output: Path) -> None:
    """
    heatmap_data: 03의 hourly_heatmap_data() 결과
                  컬럼: relationship, dow, hour, count
    """
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rels = sorted(heatmap_data["relationship"].unique(),
                  key=lambda r: list(COLORS.keys()).index(r) if r in COLORS else 99)

    fig, axs = plt.subplots(len(rels), 1, figsize=(12, 2.2 * len(rels)), sharex=True)
    if len(rels) == 1:
        axs = [axs]

    for ax, rel in zip(axs, rels):
        sub = heatmap_data[heatmap_data["relationship"] == rel]
        pivot = sub.pivot_table(index="dow", columns="hour", values="count", fill_value=0)
        pivot = pivot.reindex(dow_order).fillna(0)
        # 모든 24시간 컬럼 확보
        for h in range(24):
            if h not in pivot.columns:
                pivot[h] = 0
        pivot = pivot[sorted(pivot.columns)]

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest")
        ax.set_yticks(range(7))
        ax.set_yticklabels(dow_short, fontsize=9)
        ax.set_title(rel, fontsize=11, loc="left", fontweight="bold")

    axs[-1].set_xticks(range(0, 24, 2))
    axs[-1].set_xticklabels([f"{h}" for h in range(0, 24, 2)], fontsize=9)
    axs[-1].set_xlabel("Hour of Day", fontsize=11)

    fig.suptitle("When Do I Talk? — Temporal Patterns by Relationship",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [그림 13] {output}")


# ================================================================
# 그림 14 — 응답 시간 분포
# ================================================================
def plot_response_time(response_df: pd.DataFrame, output: Path) -> None:
    """
    response_df: 03의 response_time_distribution() 결과
                 컬럼: relationship, gap_min
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    bins = np.logspace(-1, 3, 30)  # 0.1분 ~ 1000분 로그 스케일

    for rel, sub in response_df.groupby("relationship"):
        median = sub["gap_min"].median()
        ax.hist(sub["gap_min"], bins=bins, alpha=0.55,
                label=f"{rel} (median={median:.1f}min)",
                color=COLORS.get(rel, "gray"))

    # 기준선
    ax.axvline(5, color="#2E7D32", linestyle="--", linewidth=1.5, label="5 min (immediate)")
    ax.axvline(60, color="#F4A261", linestyle="--", linewidth=1.5, label="1 hour")
    ax.axvline(360, color="#E63946", linestyle="--", linewidth=1.5, label="6 hours (delayed)")

    ax.set_xscale("log")
    ax.set_xlabel("Response Time (minutes, log scale)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Response Time Distribution — Shorter = More Intimate Relationship",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [그림 14] {output}")


# ================================================================
# 그림 15 — 종합 대시보드
# ================================================================
def plot_dashboard(metrics: pd.DataFrame, scores: pd.DataFrame, level_pct: pd.DataFrame,
                   fermi: dict, output: Path) -> None:
    """
    6-panel 종합 대시보드.
    """
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    rels = [r for r in metrics.index]
    colors = [COLORS.get(r, "gray") for r in rels]

    # (0,0) Message Volume - stacked bar
    ax = axs[0, 0]
    my_msgs = metrics["my_messages"]
    other_msgs = metrics["total_messages"] - my_msgs
    y = np.arange(len(rels))
    ax.barh(y, my_msgs, label="Me", color="#4A7A96")
    ax.barh(y, other_msgs, left=my_msgs, label="Others", color="#BDD5E0")
    ax.set_yticks(y)
    ax.set_yticklabels(rels)
    ax.set_xlabel("Total messages")
    ax.set_title("Message Volume", fontweight="bold")
    ax.legend()

    # (0,1) Reciprocity
    ax = axs[0, 1]
    ax.bar(rels, metrics["reciprocity"], color=colors)
    ax.axhline(0.5, color="#F4A261", linestyle="--", label="Balanced=0.5")
    ax.axhline(1.0, color="#2E7D32", linestyle="--", label="Perfect=1.0")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Reciprocity")
    ax.set_title("Conversation Balance", fontweight="bold")
    ax.legend(fontsize=8)
    for i, v in enumerate(metrics["reciprocity"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # (0,2) Deep Conversation Ratio
    ax = axs[0, 2]
    deep = metrics["deep_ratio_pct"]
    ax.bar(rels, deep, color=colors)
    ax.axhline(30, color="#2E7D32", linestyle="--", label="Mehl 2010 healthy ≈30%")
    ax.set_ylabel("%")
    ax.set_title("Deep Conversation Ratio (L4+L5)", fontweight="bold")
    ax.legend(fontsize=8)
    for i, v in enumerate(deep):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    # (1,0) 3-Axis Average (overall)
    ax = axs[1, 0]
    overall = scores.loc["_OVERALL_"] if "_OVERALL_" in scores.index else scores.iloc[0]
    axes_vals = [overall["Structural"], overall["Functional"], overall["Quality"]]
    axes_names = ["Structural", "Functional", "Quality"]
    bar_colors = [AXIS_COLORS[n] for n in axes_names]
    bars = ax.bar(axes_names, axes_vals, color=bar_colors)
    ax.axhline(7, color="#2E7D32", linestyle="--", linewidth=1)
    ax.axhline(4, color="#E63946", linestyle="--", linewidth=1)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Holt-Lunstad 3-Axis Average", fontweight="bold")
    for b, v in zip(bars, axes_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}",
                ha="center", fontweight="bold")

    # (1,1) Response Time
    ax = axs[1, 1]
    rt = metrics["response_median_min"]
    ax.bar(rels, rt, color=colors)
    ax.axhline(5, color="#2E7D32", linestyle="--", label="Immediate=5min")
    ax.set_ylabel("Median (min)")
    ax.set_title("Response Time", fontweight="bold")
    ax.legend(fontsize=8)
    for i, v in enumerate(rt):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

    # (1,2) Burstiness
    ax = axs[1, 2]
    ax.bar(rels, metrics["burstiness_cv"], color=colors)
    ax.axhline(1.5, color="#E63946", linestyle="--", label="Feast-or-famine >1.5")
    ax.set_ylabel("CV")
    ax.set_title("Conversation Burstiness", fontweight="bold")
    ax.legend(fontsize=8)
    for i, v in enumerate(metrics["burstiness_cv"]):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

    # 전체 타이틀과 Fermi 박스
    sci_overall = overall["SCI"]
    life_loss = fermi.get("life_loss_years", 0)
    cigs = fermi.get("cigarettes_per_day", 0)
    fig.suptitle(
        f"나의 사회적 연결성 대시보드 | My Social Connection Dashboard\n"
        f"SCI {sci_overall:.2f}/10  |  기대수명 영향 -{life_loss:.1f}yr  |  담배 {cigs:.1f}개비 등가",
        fontsize=14, fontweight="bold", y=1.02
    )

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [그림 15] {output}")


# ================================================================
# 그림 16 — Fermi 건강 영향
# ================================================================
def plot_health_impact(scenarios: pd.DataFrame, current_sci: float, output: Path) -> None:
    """
    scenarios: 05의 scenario_table() 결과
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: 시나리오별 HR 비교
    ax = axs[0]
    outcome_cols = ["HR_all_cause", "HR_cardiovascular", "HR_dementia"]
    outcome_labels = ["All-cause\nmortality", "Cardiovascular", "Dementia"]
    scenario_names = list(scenarios.index)
    scenario_colors = ["#2A9D8F", "#4A7A96", "#F4A261", "#E63946"]

    x = np.arange(len(outcome_labels))
    width = 0.2
    for i, scen in enumerate(scenario_names):
        vals = [scenarios.loc[scen, c] for c in outcome_cols]
        offset = (i - (len(scenario_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=scen, color=scenario_colors[i % len(scenario_colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8)

    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_labels)
    ax.set_ylabel("Hazard Ratio")
    ax.set_title("Predicted Health Risks by Scenario", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0.5, max(2.5, scenarios[outcome_cols].values.max() + 0.3))
    ax.grid(True, alpha=0.3, axis="y")

    # 오른쪽: SCI → 수명 손실 곡선
    ax = axs[1]
    sci_range = np.linspace(0, 10, 100)
    life_loss = [4 * (1.29 - 1.0) * (1 - s / 10) / (1 - 0 / 10) if s < 10 else 0
                 for s in sci_range]
    # 단순 선형: (1 - SCI/10) * 4년 (최악에서 4년 손실)
    life_loss = [(1 - s / 10) * 4.0 for s in sci_range]
    ax.plot(sci_range, life_loss, color="#1F4E79", linewidth=2)
    ax.fill_between(sci_range, 0, life_loss, alpha=0.2, color="#1F4E79")

    # 현재 위치 마커
    current_loss = (1 - current_sci / 10) * 4.0
    ax.axvline(current_sci, color="#E63946", linestyle="--",
               label=f"You: SCI {current_sci:.1f}\n→ {current_loss:.1f} yrs lost")
    ax.scatter([current_sci], [current_loss], color="#E63946", s=100, zorder=5)

    ax.set_xlabel("Social Connection Index (SCI)")
    ax.set_ylabel("Life Expectancy Loss (years)")
    ax.set_title("Social Connection → Life Expectancy\n(Fermi Estimation)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [그림 16] {output}")


# ================================================================
# 메인
# ================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="발표 그림 11~16 재생성")
    parser.add_argument("--messages", default="data/messages_labeled.csv")
    parser.add_argument("--metrics", default="data/relationship_metrics.csv")
    parser.add_argument("--scores", default="data/sci_scores.csv")
    parser.add_argument("--output-dir", default="figures/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모듈 내부 함수들을 재활용
    import importlib.util, sys
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    here = Path(__file__).parent
    cls_mod = _load("cls", here / "02_level_classifier.py")
    mtr_mod = _load("mtr", here / "03_metrics_per_relationship.py")
    fermi_mod = _load("fermi", here / "05_fermi_health_impact.py")

    messages = pd.read_csv(args.messages, parse_dates=["timestamp"])
    metrics = pd.read_csv(args.metrics, index_col="relationship")
    scores = pd.read_csv(args.scores, index_col=0)

    level_pct = cls_mod.level_distribution(messages)
    heatmap = mtr_mod.hourly_heatmap_data(messages)
    response = mtr_mod.response_time_distribution(messages)

    sci_overall = float(scores.loc["_OVERALL_", "SCI"]) if "_OVERALL_" in scores.index else float(scores["SCI"].mean())
    fermi_est = fermi_mod.fermi_estimate(sci_overall)
    scenarios = fermi_mod.scenario_table(sci_overall)

    print("그림 생성 중...")
    plot_three_axis_radar(scores.drop("_OVERALL_", errors="ignore"), out_dir / "fig11_radar.png")
    plot_level_distribution(level_pct, out_dir / "fig12_levels.png")
    plot_temporal_heatmap(heatmap, out_dir / "fig13_heatmap.png")
    plot_response_time(response, out_dir / "fig14_response.png")
    plot_dashboard(metrics, scores, level_pct, fermi_est, out_dir / "fig15_dashboard.png")
    plot_health_impact(scenarios, sci_overall, out_dir / "fig16_fermi.png")
    print(f"\n전체 그림 → {out_dir}/")


if __name__ == "__main__":
    main()
