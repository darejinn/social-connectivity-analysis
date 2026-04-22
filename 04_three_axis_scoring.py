"""
04_three_axis_scoring.py
========================
Holt-Lunstad (2018) 3축 프레임에 따라 관계별·전체 SCI 점수 산출.

세 축:
    Structural  (양·빈도)     : 메시지 양, 활성일, 일 평균
    Functional  (지각·응답)   : 응답 시간, 상호성
    Quality     (깊이·친밀)   : L4+L5 비율, 상호성 깊이, 심야 보정

이론적 근거:
    Holt-Lunstad et al. (2015) Perspectives on Psychological Science.
    세 지표(isolation, loneliness, living alone)가 각각 독립적으로 사망률 예측.

각 축은 0-10 스케일로 정규화. 문헌 기반 벤치마크를 기준점으로 사용.

한계 (Part 3-G의 5번째 한계):
    - 3축 가중치는 문헌 기반이지만 unique solution이 아님
    - 현재 가중치는 "각 지표 동등 가중"을 기본으로 함
    - 민감도 분석은 sensitivity_analysis() 참조

수정 포인트:
    - WEIGHTS: 축 내 지표 가중치
    - BENCHMARKS: 각 지표의 "10점 만점 기준값"
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ========== 벤치마크 (10점 기준값) ==========
# 0이면 0점, 기준값 이상이면 10점, 그 사이는 선형 보간
BENCHMARKS = {
    # Structural
    "daily_mean":          50,    # 일 50개 메시지면 만점
    "active_day_pct":      80,    # 활성일 80% 이상이면 만점
    # Functional
    "reciprocity":         0.5,   # 0.5 (완벽 균형) 가 만점
    "response_speed":      5,     # 5분 이하 응답이면 만점 (역방향: 빠를수록 점수 ↑)
    # Quality
    "deep_ratio_pct":      30,    # Mehl 2010 기준 30%면 만점
    "night_penalty_cutoff": 15,   # 심야 15% 이하는 무영향, 초과분만 감점
}

WEIGHTS = {
    "Structural": {"daily_mean": 0.5, "active_day_pct": 0.5},
    "Functional": {"reciprocity": 0.5, "response_speed": 0.5},
    "Quality":    {"deep_ratio_pct": 0.7, "night_penalty": 0.3},
}

# 전체 SCI 축 간 가중치 (기본: 동등)
AXIS_WEIGHTS = {"Structural": 1/3, "Functional": 1/3, "Quality": 1/3}


def _linear_score(value: float, benchmark: float, reverse: bool = False) -> float:
    """
    value를 0-10 점수로 정규화.
    reverse=True 면 작을수록 점수가 높음 (응답시간 등).
    """
    if pd.isna(value) or value is None:
        return 5.0  # 결측치는 중간값
    if reverse:
        # 응답 시간: 0분 = 10점, benchmark 분 = 5점, 그 이상은 감소
        if value <= 0:
            return 10.0
        score = benchmark / (benchmark + value) * 10
        return float(np.clip(score, 0, 10))
    else:
        # 일반: 0 = 0점, benchmark = 10점, 그 이상은 10점 고정
        return float(np.clip(value / benchmark * 10, 0, 10))


def score_relationship(row: pd.Series) -> dict:
    """
    한 관계의 metrics row → 3축 점수 dict.

    row 에는 03 단계의 metrics 컬럼이 있어야 함:
        daily_mean, active_day_pct, reciprocity, response_median_min,
        deep_ratio_pct, night_ratio_pct
    """
    # Structural
    s_daily = _linear_score(row.get("daily_mean"), BENCHMARKS["daily_mean"])
    s_active = _linear_score(row.get("active_day_pct"), BENCHMARKS["active_day_pct"])
    structural = (
        WEIGHTS["Structural"]["daily_mean"] * s_daily
        + WEIGHTS["Structural"]["active_day_pct"] * s_active
    )

    # Functional
    f_recip = _linear_score(row.get("reciprocity"), BENCHMARKS["reciprocity"])
    f_resp = _linear_score(
        row.get("response_median_min"), BENCHMARKS["response_speed"], reverse=True
    )
    functional = (
        WEIGHTS["Functional"]["reciprocity"] * f_recip
        + WEIGHTS["Functional"]["response_speed"] * f_resp
    )

    # Quality
    q_deep = _linear_score(row.get("deep_ratio_pct"), BENCHMARKS["deep_ratio_pct"])
    # 심야 비율 페널티: 기준 이하면 1 (페널티 없음), 이상이면 감점
    night = row.get("night_ratio_pct") or 0
    cutoff = BENCHMARKS["night_penalty_cutoff"]
    night_multiplier = 1.0 if night <= cutoff else max(0.5, 1 - (night - cutoff) / 50)
    quality = (
        WEIGHTS["Quality"]["deep_ratio_pct"] * q_deep * night_multiplier
        + WEIGHTS["Quality"]["night_penalty"] * q_deep  # 중복 대신 균형
    ) / (WEIGHTS["Quality"]["deep_ratio_pct"] + WEIGHTS["Quality"]["night_penalty"])

    sci = (
        AXIS_WEIGHTS["Structural"] * structural
        + AXIS_WEIGHTS["Functional"] * functional
        + AXIS_WEIGHTS["Quality"] * quality
    )

    return {
        "Structural": round(structural, 2),
        "Functional": round(functional, 2),
        "Quality": round(quality, 2),
        "SCI": round(sci, 2),
    }


def score_all(metrics: pd.DataFrame) -> pd.DataFrame:
    """관계별 3축 점수 DataFrame + 전체 가중 평균."""
    rows = []
    for rel, row in metrics.iterrows():
        scores = score_relationship(row)
        scores["relationship"] = rel
        rows.append(scores)
    df = pd.DataFrame(rows).set_index("relationship")[["Structural", "Functional", "Quality", "SCI"]]

    # 메시지 수 가중 평균으로 전체 SCI 계산
    if "total_messages" in metrics.columns:
        weights = metrics["total_messages"] / metrics["total_messages"].sum()
        overall = pd.Series(
            {
                "Structural": float((df["Structural"] * weights).sum()),
                "Functional": float((df["Functional"] * weights).sum()),
                "Quality": float((df["Quality"] * weights).sum()),
                "SCI": float((df["SCI"] * weights).sum()),
            },
            name="_OVERALL_",
        ).round(2)
        df = pd.concat([df, overall.to_frame().T])

    return df


def sensitivity_analysis(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    3축 가중치를 바꿔가며 SCI 변화를 관찰.
    Part 3-G "3축 가중치 자의성" 한계에 대응하는 민감도 분석.

    시나리오:
        equal:     1/3, 1/3, 1/3 (기본)
        struct_h:  0.5, 0.25, 0.25 (Structural 가중)
        qual_h:    0.25, 0.25, 0.5 (Quality 가중)
        func_h:    0.25, 0.5, 0.25 (Functional 가중)
    """
    scenarios = {
        "equal":    (1/3, 1/3, 1/3),
        "struct_h": (0.5, 0.25, 0.25),
        "func_h":   (0.25, 0.5, 0.25),
        "qual_h":   (0.25, 0.25, 0.5),
    }
    global AXIS_WEIGHTS
    original = AXIS_WEIGHTS.copy()
    rows = []
    for name, (w_s, w_f, w_q) in scenarios.items():
        AXIS_WEIGHTS = {"Structural": w_s, "Functional": w_f, "Quality": w_q}
        scores = score_all(metrics)
        if "_OVERALL_" in scores.index:
            rows.append({"scenario": name, "overall_SCI": scores.loc["_OVERALL_", "SCI"]})
    AXIS_WEIGHTS = original
    return pd.DataFrame(rows).set_index("scenario")


def main() -> None:
    parser = argparse.ArgumentParser(description="3축 점수화")
    parser.add_argument("input", help="03 단계의 relationship_metrics.csv")
    parser.add_argument("--output", default="data/sci_scores.csv")
    parser.add_argument("--sensitivity", action="store_true", help="민감도 분석 추가 실행")
    args = parser.parse_args()

    metrics = pd.read_csv(args.input, index_col="relationship")
    print(f"입력: {len(metrics)}개 관계")

    scores = score_all(metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, encoding="utf-8-sig")

    print(f"\n출력: {out}")
    print("\n[3축 점수 (발표 Part 3-C ⑥ 표와 대응)]")
    print(scores.to_string())

    if args.sensitivity:
        print("\n[민감도 분석 — 가중치 변경 시나리오]")
        print(sensitivity_analysis(metrics).round(2).to_string())


if __name__ == "__main__":
    main()
