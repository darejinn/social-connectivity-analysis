"""
05_fermi_health_impact.py
=========================
SCI 점수를 문헌의 HR(Hazard Ratio)로 환산하여 건강 영향을 Fermi 추정.

Fermi 원칙 (Part 3-A와 연계):
    1. 정확도보다 자릿수를 맞춘다.
    2. 모든 가정을 명시한다 (HR_TABLE의 출처 주석 참조).
    3. 구간으로 답하고 민감도를 분석한다 (scenarios 파라미터).

출력: 시나리오별 HR과 기대수명 손실 (년).
발표 그림 16의 원천 데이터.

문헌 HR 값:
    - All-cause mortality:  Holt-Lunstad et al. (2015) — HR 1.29 (isolation)
    - Cardiovascular:       Valtorta et al. (2016)    — RR 1.29 (CHD)
    - Dementia:             Lancet Commission (2024)  — PAF 5%
    - Depression:           Cacioppo et al. (2006)    — OR ~1.58

수정 포인트:
    - HR_TABLE: HR 값 (문헌 업데이트 시)
    - SCI_TO_RISK_SCALING: SCI → HR 매핑 함수
    - LIFE_EXPECTANCY_BASE: 한국 평균 기대수명
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ========== 문헌 HR 테이블 ==========
# (outcome, HR_at_severe_isolation, source)
HR_TABLE = {
    "all_cause_mortality": {"hr_severe": 1.29, "source": "Holt-Lunstad 2015"},
    "cardiovascular":      {"hr_severe": 1.29, "source": "Valtorta 2016 (CHD)"},
    "stroke":              {"hr_severe": 1.32, "source": "Valtorta 2016"},
    "dementia":            {"hr_severe": 1.50, "source": "Lancet 2024 (extrapolated)"},
    "depression":          {"or_severe": 1.58, "source": "Cacioppo 2006"},
}

# SCI 10 = 최적 (HR 1.00), SCI 0 = 최악 (HR = hr_severe)
# 선형 보간으로 중간값 계산
def sci_to_hr(sci: float, hr_severe: float) -> float:
    """
    SCI 0-10 → HR 선형 보간.
    SCI 10 = HR 1.00, SCI 0 = HR hr_severe.
    """
    sci_clipped = max(0, min(10, sci))
    # 10일 때 1.0, 0일 때 hr_severe
    return 1.0 + (hr_severe - 1.0) * (1 - sci_clipped / 10)


# ========== 기대수명 환산 ==========
# 한국 기대수명 (KOSIS 2023): 약 83.5세
LIFE_EXPECTANCY_BASE = 83.5

# HR → 기대수명 손실 (년)
# 대략적 변환: HR 1.1 ≈ 0.4년, HR 1.3 ≈ 1.2년 손실 (근사 선형)
# 이는 Holt-Lunstad 2010의 "담배 15개비 등가" 환산에서 유도
def hr_to_life_loss(hr: float) -> float:
    """HR → 기대수명 손실 (년). HR 1.0 = 0년, HR 1.5 = 약 2.0년."""
    if hr <= 1.0:
        return 0.0
    return (hr - 1.0) * 4.0  # 선형 근사: HR 1.25 → 1.0년


# ========== 담배 등가 ==========
# Holt-Lunstad 2010: 강한 사회관계 = 담배 15개비 금연 등가
# 따라서 SCI가 10점 떨어지면 담배 15개비 흡연 등가
def sci_to_cigarettes(sci: float) -> float:
    """SCI → 담배 등가 (개비/일)."""
    deficit = 10 - max(0, min(10, sci))
    return deficit * 1.5  # SCI 0 = 15개비, SCI 10 = 0개비


def fermi_estimate(sci: float) -> dict:
    """
    단일 SCI 값에 대한 Fermi 건강 영향 추정.

    Returns dict with:
        hr_mortality, hr_cv, hr_dementia, or_depression,
        life_loss_years, cigarettes_equivalent
    """
    return {
        "SCI": round(sci, 2),
        "HR_all_cause": round(sci_to_hr(sci, HR_TABLE["all_cause_mortality"]["hr_severe"]), 3),
        "HR_cardiovascular": round(sci_to_hr(sci, HR_TABLE["cardiovascular"]["hr_severe"]), 3),
        "HR_dementia": round(sci_to_hr(sci, HR_TABLE["dementia"]["hr_severe"]), 3),
        "OR_depression": round(sci_to_hr(sci, HR_TABLE["depression"]["or_severe"]), 3),
        "life_loss_years": round(hr_to_life_loss(sci_to_hr(sci, 1.29)), 2),
        "cigarettes_per_day": round(sci_to_cigarettes(sci), 1),
    }


def scenario_table(
    current_sci: float,
    optimal_sci: float = 10.0,
    moderate_isolation_sci: float = 4.0,
    severe_isolation_sci: float = 1.0,
) -> pd.DataFrame:
    """
    발표 그림 16의 4개 시나리오 비교 표를 생성.
    Optimal / Current / Moderate Isolation / Severe Isolation.
    """
    scenarios = {
        "Optimal (SCI 10)": optimal_sci,
        f"Your Current (SCI {current_sci:.1f})": current_sci,
        f"Moderate Isolation (SCI {moderate_isolation_sci:.0f})": moderate_isolation_sci,
        f"Severe Isolation (SCI {severe_isolation_sci:.0f})": severe_isolation_sci,
    }
    rows = []
    for name, sci in scenarios.items():
        est = fermi_estimate(sci)
        est["scenario"] = name
        rows.append(est)
    df = pd.DataFrame(rows).set_index("scenario")
    col_order = [
        "SCI",
        "HR_all_cause", "HR_cardiovascular", "HR_dementia", "OR_depression",
        "life_loss_years", "cigarettes_per_day",
    ]
    return df[col_order]


def sensitivity_band(sci: float, noise_pct: float = 0.4) -> pd.DataFrame:
    """
    SCI의 ±noise_pct 범위로 Fermi 추정의 구간 제공.
    Part 3-G의 "Quality 점수 ±40% 변동" 한계에 대응.
    """
    low = sci * (1 - noise_pct)
    high = min(10, sci * (1 + noise_pct))
    df = pd.DataFrame({
        "worst":   fermi_estimate(low),
        "central": fermi_estimate(sci),
        "best":    fermi_estimate(high),
    }).T
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fermi 건강 영향 추정")
    parser.add_argument("input", nargs="?", help="04 단계의 sci_scores.csv (없으면 SCI 직접 입력)")
    parser.add_argument("--sci", type=float, help="SCI 값 직접 입력 (sci_scores.csv 없을 때)")
    parser.add_argument("--output", default="data/health_impact.csv")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, index_col=0)
        if "_OVERALL_" in df.index:
            sci = float(df.loc["_OVERALL_", "SCI"])
        else:
            sci = float(df["SCI"].mean())
        print(f"입력 SCI (overall): {sci:.2f}")
    elif args.sci is not None:
        sci = args.sci
        print(f"입력 SCI (직접): {sci:.2f}")
    else:
        parser.error("input 파일 또는 --sci 필수")

    table = scenario_table(sci)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, encoding="utf-8-sig")

    print(f"\n출력: {out}")
    print("\n[Fermi 시나리오 비교 (그림 16의 원천 데이터)]")
    print(table.to_string())

    if args.sensitivity:
        print("\n[민감도 대역 (±40%) — Part 3-G 대응]")
        print(sensitivity_band(sci).to_string())


if __name__ == "__main__":
    main()
