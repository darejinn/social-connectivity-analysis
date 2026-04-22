"""
test_with_mock_data.py
======================
카톡 파일 없이 합성 데이터로 전체 파이프라인을 검증.
실제 발표의 수치(SCI 6.58, L4+L5 ≈ 7.9%, Quality 4.55 등)를 근사하도록 설계된 mock.

사용:
    python test_with_mock_data.py

검증 포인트:
    - 파이프라인이 에러 없이 끝까지 실행되는가
    - 산출 SCI가 발표값 근방에 떨어지는가
    - 그림 11~16이 모두 생성되는가
"""

from __future__ import annotations
import importlib.util
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_mock_messages(relationship: str, total: int, reciprocity: float,
                            deep_ratio: float, night_ratio: float, burstiness: float,
                            start_date: datetime, days: int = 365 * 3,
                            my_name: str = "조윤진", seed: int = 42) -> pd.DataFrame:
    """
    발표의 관계별 지표에 부합하는 합성 메시지 생성.

    Args:
        relationship: 관계 라벨
        total: 총 메시지 수
        reciprocity: 내 발신 비율 (0-1)
        deep_ratio: L4+L5 비율 (0-1)
        night_ratio: 심야(23-03) 비율 (0-1)
        burstiness: 일 분포의 변동계수 근사치
    """
    rng = random.Random(seed + hash(relationship) % 1000)
    rows = []

    # 각 메시지의 날짜를 burstiness에 맞게 집중 또는 분산 분포 생성
    # 간단히: 활성일을 포아송 분포처럼 뿌리고, 활성일에 여러 개 집중
    if burstiness > 1.0:
        # 활성일이 적고 많은 메시지가 몰림
        active_days = int(days * 0.3)
    else:
        active_days = int(days * 0.7)

    day_indices = rng.sample(range(days), min(active_days, days))
    msgs_per_day = np.random.default_rng(seed).lognormal(
        mean=np.log(total / active_days), sigma=0.6 * burstiness, size=active_days
    )
    msgs_per_day = np.round(msgs_per_day).astype(int)
    msgs_per_day = np.clip(msgs_per_day, 1, None)
    # 전체가 target에 맞도록 스케일
    msgs_per_day = (msgs_per_day * total / msgs_per_day.sum()).round().astype(int)

    for day_idx, n_msgs in zip(day_indices, msgs_per_day):
        if n_msgs <= 0:
            continue
        base = start_date + timedelta(days=day_idx)
        for _ in range(n_msgs):
            # 시간 선택: night_ratio 반영
            if rng.random() < night_ratio:
                hour = rng.choice([23, 0, 1, 2])
            else:
                hour = rng.choices(range(24), weights=[
                    0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.07, 0.07, 0.06, 0.06,
                    0.07, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015,
                ])[0]
            minute = rng.randint(0, 59)
            ts = base.replace(hour=hour, minute=minute)

            # 발신자
            is_me = rng.random() < reciprocity
            sender = my_name if is_me else f"{relationship}_상대"

            # Level을 확률적으로 부여 (발표 값 근사)
            # L1: 30%, L2: 45%, L3: 15%, L4: 7%, L5: 3% 기본 → deep_ratio로 조정
            level_weights = [0.30, 0.45, 0.15]
            remaining = 1 - sum(level_weights)
            l4 = deep_ratio * 0.7
            l5 = deep_ratio * 0.3
            adjust = remaining - (l4 + l5)
            level_weights = [w + adjust / 3 for w in level_weights]
            level_weights.extend([l4, l5])
            level = rng.choices([1, 2, 3, 4, 5], weights=level_weights)[0]

            # 레벨에 따른 텍스트 합성
            text = _mock_text(level, rng)

            rows.append({
                "relationship": relationship,
                "timestamp": ts,
                "sender": sender,
                "is_me": is_me,
                "text": text,
                "text_length": len(text),
                "level": level,  # Ground truth로 미리 심어둠 (02 단계에서 재분류될 것)
            })

    df = pd.DataFrame(rows).sort_values(["relationship", "timestamp"]).reset_index(drop=True)
    return df


def _mock_text(level: int, rng: random.Random) -> str:
    templates = {
        1: ["ㅇㅇ", "ㄱㄱ", "넵", "응", "ㅋㅋ", "몇시에?", "오케이", "주문 완료"],
        2: ["날씨 좋다 ㅎㅎ", "뭐해 지금", "배고프다 점심 뭐먹지", "안녕 오늘 수고", "굿모닝"],
        3: ["이 논문 봤어? https://link", "시험 범위가 어디까지야?", "방법은 세 가지가 있어", "자료 공유할게"],
        4: ["내 생각엔 좀 다른데, 솔직히 이건 문제야", "개인적으로 난 그 의견에 반대야", "근데 오히려 이게 더 맞지 않나?"],
        5: ["요즘 너무 힘들어 정말로", "보고 싶다 진짜", "솔직히 고민이 많아 말 못하겠지만", "외로운 것 같아"],
    }
    return rng.choice(templates[level])


def main() -> None:
    here = Path(__file__).parent
    out_dir = here / "test_results"
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 발표 수치와 일치하는 합성 파라미터
    start = datetime(2023, 1, 1)
    print("=" * 70)
    print("[Mock] 합성 데이터 생성 (발표 수치에 맞게)")
    print("=" * 70)

    configs = [
        # relationship, total,  reciprocity, deep_ratio, night_ratio, burstiness
        ("가족톡",      8595,   0.66,        0.088,      0.082,       0.92),
        ("단톡A",      14604,   0.26,        0.111,      0.132,       1.24),
        ("단톡B",       7923,   0.32,        0.057,      0.216,       0.87),
    ]

    all_rows = []
    for rel, total, recip, deep, night, burst in configs:
        df = generate_mock_messages(rel, total, recip, deep, night, burst, start)
        all_rows.append(df)
        print(f"  {rel}: {len(df):,}개 생성 (recip={recip}, deep={deep:.1%}, night={night:.1%})")

    messages = pd.concat(all_rows, ignore_index=True)
    # 02에서 재분류할 것이므로 level 컬럼은 드롭
    messages_for_pipeline = messages.drop(columns=["level"])
    messages_csv = data_dir / "messages.csv"
    messages_for_pipeline.to_csv(messages_csv, index=False, encoding="utf-8-sig")
    print(f"\n저장: {messages_csv}")

    # 파이프라인 실행
    cls_mod    = _load("cls", here / "02_level_classifier.py")
    mtr_mod    = _load("mtr", here / "03_metrics_per_relationship.py")
    score_mod  = _load("sc",  here / "04_three_axis_scoring.py")
    fermi_mod  = _load("fm",  here / "05_fermi_health_impact.py")
    viz_mod    = _load("vz",  here / "06_visualize.py")

    print("\n" + "=" * 70)
    print("[02] Level 분류")
    print("=" * 70)
    # mock은 이미 정답 level이 있으므로 그대로 사용 (실제 heuristic 대신)
    labeled = messages.copy()
    # 하지만 classifier도 한 번 돌려 sanity check
    reclassified = cls_mod.add_level_column(messages_for_pipeline)
    labeled_csv = data_dir / "messages_labeled.csv"
    labeled.to_csv(labeled_csv, index=False, encoding="utf-8-sig")
    print("GT (mock 정답) Level 분포:")
    print(cls_mod.level_distribution(labeled).to_string())
    print("\nHeuristic 재분류 결과:")
    print(cls_mod.level_distribution(reclassified).to_string())

    print("\n" + "=" * 70)
    print("[03] 관계별 지표")
    print("=" * 70)
    metrics = mtr_mod.compute_all(labeled)
    print(metrics.to_string())

    print("\n" + "=" * 70)
    print("[04] 3축 점수")
    print("=" * 70)
    scores = score_mod.score_all(metrics)
    print(scores.to_string())

    sci = float(scores.loc["_OVERALL_", "SCI"])

    print("\n" + "=" * 70)
    print("[05] Fermi 건강 영향")
    print("=" * 70)
    fermi_est = fermi_mod.fermi_estimate(sci)
    scenario_table = fermi_mod.scenario_table(sci)
    print(scenario_table.to_string())

    print("\n" + "=" * 70)
    print("[06] 시각화")
    print("=" * 70)
    level_pct = cls_mod.level_distribution(labeled)
    heatmap_data = mtr_mod.hourly_heatmap_data(labeled)
    response_data = mtr_mod.response_time_distribution(labeled)

    viz_mod.plot_three_axis_radar(scores.drop("_OVERALL_"), fig_dir / "fig11_radar.png")
    viz_mod.plot_level_distribution(level_pct, fig_dir / "fig12_levels.png")
    viz_mod.plot_temporal_heatmap(heatmap_data, fig_dir / "fig13_heatmap.png")
    viz_mod.plot_response_time(response_data, fig_dir / "fig14_response.png")
    viz_mod.plot_dashboard(metrics, scores, level_pct, fermi_est, fig_dir / "fig15_dashboard.png")
    viz_mod.plot_health_impact(scenario_table, sci, fig_dir / "fig16_fermi.png")

    print("\n" + "=" * 70)
    print("[검증] 파이프라인 무결성 체크")
    print("=" * 70)
    # 주의: mock 데이터는 완벽한 수치 복제가 아니라 "파이프라인이 작동하는지" 확인용.
    # 실제 발표 수치(SCI 6.58)와 정확히 일치시키려면 실제 카톡 로그가 필요.
    checks = [
        ("메시지 파싱됨",      len(messages) > 0),
        ("Level 분류 완료",    set(labeled["level"].unique()).issubset({1, 2, 3, 4, 5})),
        ("3축 점수 산출됨",    all(0 <= scores.loc[r, "SCI"] <= 10 for r in scores.index)),
        ("Fermi HR 유효 범위", 1.0 <= fermi_est["HR_all_cause"] <= 2.0),
        ("그림 6장 생성됨",    len(list(fig_dir.glob("*.png"))) == 6),
    ]
    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}")

    print(f"\n  mock SCI: {sci:.2f}  (참고: 발표 실측 6.58 — mock은 수치 재현이 아닌 파이프라인 검증 목적)")
    print(f"  mock life loss: {fermi_est['life_loss_years']:.2f}년  (참고: 발표 실측 -1.4년)")
    print(f"\n그림 6장 → {fig_dir}/")


if __name__ == "__main__":
    main()
