"""
03_metrics_per_relationship.py
==============================
관계(가족톡·단톡A·단톡B)별로 발표에 등장하는 모든 정량 지표 계산.

발표 Part 3-C의 "관계별 종합 지표" 표를 이 스크립트로 재현.

계산되는 지표:
    1. 메시지 총량 (total_messages)
    2. 상호성 (reciprocity): 내 발신 ÷ (내 발신 + 타인 발신). 0.5면 균형.
    3. 응답 시간 중앙값 (response_median_min): 상대방 메시지 직후 내 응답까지 분
    4. 깊은 대화 비율 (deep_ratio_pct): L4+L5 비율
    5. 심야 비율 (night_ratio_pct): 23시 ~ 03시 메시지 비율
    6. 활성일 비율 (active_day_pct): 전체 관측 기간 중 메시지 있는 날 비율
    7. Burstiness (CV): 일 메시지 수의 변동계수 = std/mean
    8. 일 평균 메시지 (daily_mean)

한계:
    - 응답 시간은 "내 직전 메시지가 있으면" 제외 (연속 내 발신은 응답이 아님)
    - 단톡방에서는 "누구에게의 응답"을 추정하기 어려워 전체 흐름 기준
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NIGHT_HOURS = set(list(range(23, 24)) + list(range(0, 3)))  # 23, 0, 1, 2시


def compute_metrics_one(df_rel: pd.DataFrame) -> dict:
    """단일 관계(DataFrame) 에 대한 지표 dict 반환."""
    n = len(df_rel)
    if n == 0:
        return {}

    df = df_rel.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. 메시지 총량
    total = n

    # 2. 상호성
    my_count = int(df["is_me"].sum())
    reciprocity = my_count / total if total else 0.0

    # 3. 응답 시간 (분)
    #    바로 직전 메시지가 타인이고 현재가 내 메시지인 경우의 시간 차이
    df["prev_is_me"] = df["is_me"].shift(1)
    df["prev_ts"] = df["timestamp"].shift(1)
    df["gap_min"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 60
    response_mask = (df["is_me"] == True) & (df["prev_is_me"] == False) & (df["gap_min"].notna())
    # 24시간 초과 응답은 제외 (대화 종료로 간주)
    valid_resp = df.loc[response_mask & (df["gap_min"] < 60 * 24), "gap_min"]
    resp_median = float(valid_resp.median()) if len(valid_resp) else np.nan
    resp_mean = float(valid_resp.mean()) if len(valid_resp) else np.nan

    # 4. 깊은 대화 비율
    if "level" in df.columns:
        deep = float(((df["level"] == 4) | (df["level"] == 5)).mean() * 100)
    else:
        deep = np.nan

    # 5. 심야 비율
    df["hour"] = df["timestamp"].dt.hour
    night = float(df["hour"].isin(NIGHT_HOURS).mean() * 100)

    # 6. 활성일 비율
    df["date"] = df["timestamp"].dt.date
    first_day = df["date"].min()
    last_day = df["date"].max()
    total_days = (last_day - first_day).days + 1
    active_days = df["date"].nunique()
    active_pct = float(active_days / total_days * 100) if total_days > 0 else 0.0

    # 7. Burstiness (일 메시지 수의 CV)
    daily = df.groupby("date").size()
    # 0인 날도 포함하기 위해 reindex
    all_days = pd.date_range(first_day, last_day, freq="D").date
    daily = daily.reindex(all_days, fill_value=0)
    mean = daily.mean()
    std = daily.std()
    burstiness = float(std / mean) if mean > 0 else np.nan
    daily_mean = float(mean)

    return {
        "total_messages": total,
        "my_messages": my_count,
        "reciprocity": round(reciprocity, 3),
        "response_median_min": round(resp_median, 2) if not np.isnan(resp_median) else None,
        "response_mean_min": round(resp_mean, 2) if not np.isnan(resp_mean) else None,
        "deep_ratio_pct": round(deep, 1) if not np.isnan(deep) else None,
        "night_ratio_pct": round(night, 1),
        "active_day_pct": round(active_pct, 1),
        "burstiness_cv": round(burstiness, 2) if not np.isnan(burstiness) else None,
        "daily_mean": round(daily_mean, 1),
        "total_days": total_days,
        "active_days": active_days,
    }


def hourly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """관계별 × 요일별 × 시간별 메시지 카운트. 그림 13 히트맵의 원천 데이터."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["dow"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour

    grouped = (
        df.groupby(["relationship", "dow", "hour"])
          .size()
          .reset_index(name="count")
    )
    return grouped


def response_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """관계별 응답 시간 리스트 (분). 그림 14 히스토그램의 원천 데이터."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["relationship", "timestamp"]).reset_index(drop=True)

    # 관계 내에서만 shift
    df["prev_is_me"] = df.groupby("relationship")["is_me"].shift(1)
    df["prev_ts"] = df.groupby("relationship")["timestamp"].shift(1)
    df["gap_min"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 60

    mask = (
        (df["is_me"] == True)
        & (df["prev_is_me"] == False)
        & (df["gap_min"].notna())
        & (df["gap_min"] < 60 * 24)
    )
    return df.loc[mask, ["relationship", "gap_min"]]


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """전체 관계별 metrics DataFrame을 반환."""
    rows = []
    for rel, sub in df.groupby("relationship"):
        metrics = compute_metrics_one(sub)
        metrics["relationship"] = rel
        rows.append(metrics)
    out = pd.DataFrame(rows).set_index("relationship")
    # 발표 표 순서에 맞게 컬럼 정렬
    col_order = [
        "total_messages", "my_messages", "reciprocity",
        "response_median_min", "response_mean_min",
        "deep_ratio_pct", "night_ratio_pct", "active_day_pct",
        "burstiness_cv", "daily_mean", "total_days", "active_days",
    ]
    return out[[c for c in col_order if c in out.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(description="관계별 지표 계산")
    parser.add_argument("input", help="02 단계의 messages_labeled.csv")
    parser.add_argument("--output", default="data/relationship_metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"입력: {len(df):,}개 메시지, {df['relationship'].nunique()}개 관계")

    metrics = compute_all(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out, encoding="utf-8-sig")

    print(f"\n출력: {out}")
    print("\n[관계별 지표 (발표 Part 3-C ⑤ 표와 대응)]")
    print(metrics.to_string())


if __name__ == "__main__":
    main()
