"""
run_all.py
==========
전체 분석 파이프라인을 일괄 실행.

사용법:
    python run_all.py \\
        --inputs data/KakaoTalk_가족톡.txt data/KakaoTalk_단톡A.txt data/KakaoTalk_단톡B.txt \\
        --names 가족톡 단톡A 단톡B \\
        --my-name 조윤진 \\
        --output-dir results/

실행 순서:
    01. 파싱          → messages.csv
    02. Level 분류    → messages_labeled.csv
    03. 관계별 지표   → relationship_metrics.csv
    04. 3축 점수      → sci_scores.csv
    05. Fermi 건강    → health_impact.csv
    06. 시각화        → figures/fig11~16.png
"""

from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path


def _load(name: str, path: Path):
    """같은 폴더의 숫자 prefix 모듈을 import."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="전체 분석 파이프라인 일괄 실행")
    parser.add_argument("--inputs", nargs="+", required=True, help="카톡 .txt 파일 경로들")
    parser.add_argument("--names", nargs="+", required=True, help="각 입력의 관계 라벨")
    parser.add_argument("--my-name", default="조윤진", help="본인 카톡 닉네임")
    parser.add_argument("--output-dir", default="results/", help="전체 산출물 디렉토리")
    args = parser.parse_args()

    if len(args.inputs) != len(args.names):
        parser.error("inputs와 names의 개수가 같아야 합니다")

    here = Path(__file__).parent
    out = Path(args.output_dir)
    data_dir = out / "data"
    fig_dir = out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 모듈 동적 import
    parser_mod = _load("p01", here / "01_parse_kakaotalk.py")
    cls_mod    = _load("p02", here / "02_level_classifier.py")
    mtr_mod    = _load("p03", here / "03_metrics_per_relationship.py")
    score_mod  = _load("p04", here / "04_three_axis_scoring.py")
    fermi_mod  = _load("p05", here / "05_fermi_health_impact.py")
    viz_mod    = _load("p06", here / "06_visualize.py")

    # =====================
    # 01. 파싱
    # =====================
    print("=" * 70)
    print("[01] 카카오톡 파싱")
    print("=" * 70)
    files = {name: Path(p) for name, p in zip(args.names, args.inputs)}
    messages = parser_mod.parse_multiple(files, my_name=args.my_name)
    messages_csv = data_dir / "messages.csv"
    messages.to_csv(messages_csv, index=False, encoding="utf-8-sig")
    print(f"  → {messages_csv}  ({len(messages):,}개 메시지)")

    # =====================
    # 02. Level 분류
    # =====================
    print("\n" + "=" * 70)
    print("[02] Level 분류 (L1~L5)")
    print("=" * 70)
    labeled = cls_mod.add_level_column(messages)
    labeled_csv = data_dir / "messages_labeled.csv"
    labeled.to_csv(labeled_csv, index=False, encoding="utf-8-sig")
    level_pct = cls_mod.level_distribution(labeled)
    print(level_pct.to_string())
    print(f"  → {labeled_csv}")

    # =====================
    # 03. 관계별 지표
    # =====================
    print("\n" + "=" * 70)
    print("[03] 관계별 지표")
    print("=" * 70)
    metrics = mtr_mod.compute_all(labeled)
    metrics_csv = data_dir / "relationship_metrics.csv"
    metrics.to_csv(metrics_csv, encoding="utf-8-sig")
    print(metrics.to_string())

    # =====================
    # 04. 3축 점수
    # =====================
    print("\n" + "=" * 70)
    print("[04] Holt-Lunstad 3축 점수")
    print("=" * 70)
    scores = score_mod.score_all(metrics)
    scores_csv = data_dir / "sci_scores.csv"
    scores.to_csv(scores_csv, encoding="utf-8-sig")
    print(scores.to_string())

    # =====================
    # 05. Fermi 건강
    # =====================
    print("\n" + "=" * 70)
    print("[05] Fermi 건강 영향")
    print("=" * 70)
    sci_overall = float(scores.loc["_OVERALL_", "SCI"]) if "_OVERALL_" in scores.index else float(scores["SCI"].mean())
    sci_overall = min(10.0, max(0.0, sci_overall))
    fermi_est = fermi_mod.fermi_estimate(sci_overall)
    scenario_table = fermi_mod.scenario_table(sci_overall)
    health_csv = data_dir / "health_impact.csv"
    scenario_table.to_csv(health_csv, encoding="utf-8-sig")
    print(scenario_table.to_string())
    print(f"\n  전체 SCI: {sci_overall:.2f}  →  기대수명 영향 -{fermi_est['life_loss_years']:.2f}년")
    print(f"  담배 등가: {fermi_est['cigarettes_per_day']:.1f}개비/일")

    # =====================
    # 06. 시각화
    # =====================
    print("\n" + "=" * 70)
    print("[06] 시각화 (그림 11~16)")
    print("=" * 70)

    heatmap_data = mtr_mod.hourly_heatmap_data(labeled)
    response_data = mtr_mod.response_time_distribution(labeled)

    viz_mod.plot_three_axis_radar(scores.drop("_OVERALL_", errors="ignore"), fig_dir / "fig11_radar.png")
    viz_mod.plot_level_distribution(level_pct, fig_dir / "fig12_levels.png")
    viz_mod.plot_temporal_heatmap(heatmap_data, fig_dir / "fig13_heatmap.png")
    viz_mod.plot_response_time(response_data, fig_dir / "fig14_response.png")
    viz_mod.plot_dashboard(metrics, scores, level_pct, fermi_est, fig_dir / "fig15_dashboard.png")
    viz_mod.plot_health_impact(scenario_table, sci_overall, fig_dir / "fig16_fermi.png")

    # =====================
    # 요약
    # =====================
    print("\n" + "=" * 70)
    print("[완료] 파이프라인 전체 실행 완료")
    print("=" * 70)
    print(f"데이터:  {data_dir}/")
    print(f"그림:    {fig_dir}/")
    print(f"\n최종 SCI: {sci_overall:.2f}/10")
    print(f"기대수명 영향: -{fermi_est['life_loss_years']:.2f}년")
    print(f"담배 등가:     {fermi_est['cigarettes_per_day']:.1f}개비/일")


if __name__ == "__main__":
    main()
