"""
02_level_classifier.py
======================
카톡 메시지를 L1 (거래) ~ L5 (감정) 다섯 레벨로 분류하는 규칙 기반 heuristic.

이론적 근거: Mehl et al. (2010) Psychological Science의 "substantive conversation"
개념을 이원(small talk vs substantive) 분류에서 5단계로 확장.

Level 정의:
    L1 거래 (Transactional):   기능적 주고받기, 단답, 이모티콘만, 약속·주문 등
    L2 잡담 (Small talk):      일상 얕은 대화, 안부, 날씨, 가벼운 유머
    L3 정보 (Informational):   구체적 정보 공유, 링크, 뉴스, 질문 답변
    L4 의견 (Opinion):         평가·판단·의견 표명, "나는 이렇게 생각해"
    L5 감정 (Emotional):       감정·내면 공유, 고민, 취약성 노출

Mehl 2010 건강 기준: substantive 대화 (L3+L4+L5) 비율 약 30%.
본 발표에서는 L4+L5를 "깊은 대화"의 proxy로 사용.

한계 (Part 3-G와 일치):
    - 규칙 기반 heuristic이므로 전문가 라벨 대비 ~75% 일치 추정
    - 맥락·반어·암묵적 감정 놓침
    - 개선: LLM 분류기 (GPT-4o, Claude) 사용 시 ~90% 도달 예상

수정 포인트:
    - classify_level(): 분류 규칙 전체
    - 각 레벨의 키워드 리스트 (L1_PATTERNS 등)
    - MEHL_HEALTHY_RATIO: 건강 기준치
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd


# ========== 설정 ==========
MEHL_HEALTHY_RATIO = 0.30  # Mehl 2010 substantive 기준

# ========== 패턴 정의 ==========

# L1 (거래) — 단답, 승낙·거절, 주문·약속
L1_PATTERNS = {
    "single_token": re.compile(r"^(ㅇㅇ|ㅇㅋ|ㄱㄱ|ㄴㄴ|응|넵|네|예|아니|오케|오키|ㅇㅇㅇ|굿|굳)$"),
    "emoji_only": re.compile(r"^[ㅋㅎㅜㅠ\s!?.\U0001F300-\U0001FAFF\u2600-\u27BF]+$"),
    "transaction": re.compile(r"(주문|결제|송금|입금|예약|취소|영수증|배송|도착|픽업)"),
    "scheduling": re.compile(r"(몇 ?시|어디서|언제|시간 ?돼|약속)"),
}

# L2 (잡담) — 안부, 날씨, 가벼운 웃음
L2_PATTERNS = {
    "greeting": re.compile(r"(안녕|굿모닝|잘 ?자|굿나잇|수고|좋은 ?하루)"),
    "weather": re.compile(r"(날씨|춥|덥|비 ?오|눈 ?오)"),
    "food_mundane": re.compile(r"(뭐 ?먹|점심|저녁|아침|배고프|맛있)"),
    "mundane": re.compile(r"(ㅋㅋㅋ|ㅎㅎ|그냥|뭐해|심심)"),
}

# L3 (정보) — 질문·답변, 링크, 구체 정보
L3_PATTERNS = {
    "url": re.compile(r"https?://|www\."),
    "question": re.compile(r"\?(?!\?)"),  # 물음표 (중복 제외)
    "info_keywords": re.compile(r"(참고|자료|논문|시험|수업|과제|공부|방법|어떻게|왜)"),
    "explanation": re.compile(r"(왜냐하면|그래서|이유|때문에|즉|다시 ?말해)"),
}

# L4 (의견) — 평가, 주장, 가치 판단
L4_PATTERNS = {
    "opinion": re.compile(r"(내 ?생각|개인적으로|내 ?의견|나는|난 |내가 ?보기|솔직히)"),
    "evaluation": re.compile(r"(좋다|나쁘다|별로|괜찮|이상해|문제|맞|틀렸|동의|반대)"),
    "argument": re.compile(r"(하지만|그러나|반면|오히려|사실은|근데)"),
}

# L5 (감정) — 감정 토로, 취약성, 관계 자체 언급
L5_PATTERNS = {
    "emotion_strong": re.compile(
        r"(사랑|보고 ?싶|그리워|외로|슬프|우울|힘들|지쳤|괴로|두려워|무서워|불안)"
    ),
    "vulnerability": re.compile(r"(고민|걱정|털어놓|솔직히 ?말해|사실 ?나)"),
    "relational": re.compile(r"(우리 ?사이|관계|친구로서|너와 ?나|함께 ?한)"),
    "gratitude_deep": re.compile(r"(정말 ?고마|진심으로|너무 ?감사|덕분에)"),
}


def classify_level(text: str) -> int:
    """
    단일 메시지의 Level 판정.

    판정 우선순위: L5 > L4 > L3 > L2 > L1 > L2 (기본값)
    즉 감정·의견이 있으면 해당 레벨로 우선 분류.

    Args:
        text: 메시지 텍스트

    Returns:
        1~5 정수
    """
    if not text or not text.strip():
        return 1

    t = text.strip()

    # L5: 감정·취약성 (최우선)
    if any(p.search(t) for p in L5_PATTERNS.values()):
        return 5

    # L4: 의견·평가
    if any(p.search(t) for p in L4_PATTERNS.values()):
        # 단, 짧은 단답(L1)이면 L1 우선
        if len(t) < 10 and L1_PATTERNS["single_token"].match(t):
            return 1
        return 4

    # L1: 단답·거래 (짧은 것 먼저 걸러냄)
    if len(t) < 15:
        if L1_PATTERNS["single_token"].match(t):
            return 1
        if L1_PATTERNS["emoji_only"].match(t):
            return 1
    if L1_PATTERNS["transaction"].search(t) or L1_PATTERNS["scheduling"].search(t):
        return 1

    # L3: 정보·질문·링크
    if any(p.search(t) for p in L3_PATTERNS.values()):
        return 3

    # L2: 잡담 (기본값에 가까움)
    if any(p.search(t) for p in L2_PATTERNS.values()):
        return 2

    # 길이 기반 보조 판정
    if len(t) > 80:
        return 3  # 긴 메시지는 정보성으로 추정
    return 2  # 기본값: 잡담


def add_level_column(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 level 컬럼 추가. 빈 텍스트와 사진·이모티콘 전용 메시지 처리 포함."""
    df = df.copy()
    df["text"] = df["text"].fillna("")

    # 카톡이 "사진", "이모티콘" 등으로 남긴 시스템 메시지는 L1로 분류
    system_msgs = df["text"].str.match(r"^(사진|동영상|이모티콘|파일: |음성메시지)")

    df["level"] = df["text"].apply(classify_level)
    df.loc[system_msgs, "level"] = 1
    return df


def level_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """관계별 × Level별 분포 (%)."""
    counts = (
        df.groupby(["relationship", "level"])
          .size()
          .unstack(fill_value=0)
    )
    # 누락된 레벨 컬럼을 0으로 채움
    for lv in range(1, 6):
        if lv not in counts.columns:
            counts[lv] = 0
    counts = counts[[1, 2, 3, 4, 5]]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct.columns = [f"L{c}" for c in pct.columns]
    return pct.round(1)


def substantive_ratio(df: pd.DataFrame) -> pd.Series:
    """관계별 '깊은 대화 (L4+L5)' 비율."""
    pct = level_distribution(df)
    return (pct["L4"] + pct["L5"]) / 100  # 0-1 스케일


def main() -> None:
    parser = argparse.ArgumentParser(description="메시지에 L1~L5 level 부여")
    parser.add_argument("input", help="01 단계의 messages.csv 경로")
    parser.add_argument("--output", default="data/messages_labeled.csv", help="출력 경로")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"입력: {len(df):,}개 메시지")

    df = add_level_column(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"출력: {out}")

    print("\n[관계별 Level 분포 (%)]")
    print(level_distribution(df).to_string())

    print("\n[깊은 대화 (L4+L5) 비율]")
    print(f"Mehl 2010 건강 기준: {MEHL_HEALTHY_RATIO * 100:.0f}%")
    print(substantive_ratio(df).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
