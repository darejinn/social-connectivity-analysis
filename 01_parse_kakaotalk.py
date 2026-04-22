"""
01_parse_kakaotalk.py
=====================
카카오톡 PC/Mac 버전에서 내보낸 .txt 대화 로그를 구조화된 DataFrame으로 파싱.

카카오톡 내보내기 형식 (PC):
    [이름] [오후 3:42] 메시지 내용
    또는
    이름 오후 3:42 메시지 내용

날짜는 "--------------- 2024년 3월 15일 금요일 ---------------" 같은 헤더로 등장.

입력:  data/KakaoTalk_가족톡.txt (등)
출력:  data/messages.csv
       컬럼: relationship, timestamp, sender, is_me, text, text_length

수정 포인트:
    - MY_NAME: 본인 카톡 닉네임 (정확히 일치해야 is_me 플래그가 맞음)
    - DATE_HEADER_RE: 내보내기 형식이 다르면 조정
    - MESSAGE_RE: 메시지 라인 정규식
"""

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

# ========== 사용자 설정 ==========
MY_NAME = "조윤진"  # 카톡 내 본인 닉네임. 본인 환경에 맞게 수정 필요.

# 카톡 PC 버전의 날짜 헤더: "--------------- 2024년 3월 15일 금요일 ---------------"
DATE_HEADER_RE = re.compile(
    r"^-+\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일.*-+\s*$"
)

# 메시지 라인:  [이름] [오후 3:42] 메시지
MESSAGE_RE = re.compile(
    r"^\[(?P<sender>[^\]]+)\]\s*\[(?P<ampm>오전|오후)\s*(?P<h>\d{1,2}):(?P<m>\d{2})\]\s*(?P<text>.*)$"
)

# Mac/모바일 변형:  이름 오후 3:42 : 메시지
MESSAGE_RE_ALT = re.compile(
    r"^(?P<sender>[^,]+),\s*(?P<ampm>오전|오후)\s*(?P<h>\d{1,2}):(?P<m>\d{2})\s*:\s*(?P<text>.*)$"
)


@dataclass
class ParsedMessage:
    relationship: str
    timestamp: datetime
    sender: str
    is_me: bool
    text: str

    def to_dict(self) -> dict:
        return {
            "relationship": self.relationship,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "is_me": self.is_me,
            "text": self.text,
            "text_length": len(self.text),
        }


def _to_24h(ampm: str, h: int, m: int) -> tuple[int, int]:
    """카톡의 오전/오후 표기를 24시간제로 변환."""
    if ampm == "오전":
        return (0 if h == 12 else h, m)
    else:  # 오후
        return (h if h == 12 else h + 12, m)


def parse_kakaotalk(path: Path, relationship: str, my_name: str = MY_NAME) -> list[ParsedMessage]:
    """
    카카오톡 .txt 로그 한 개를 파싱.

    Args:
        path: .txt 파일 경로
        relationship: 이 로그에 붙일 관계 라벨 ("가족톡", "단톡A", "단톡B" 등)
        my_name: 본인 닉네임 (is_me 판정용)

    Returns:
        ParsedMessage 리스트
    """
    messages: list[ParsedMessage] = []
    current_date: datetime | None = None
    last_msg: ParsedMessage | None = None  # 연속 라인 병합용

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            # 날짜 헤더
            m_date = DATE_HEADER_RE.match(line)
            if m_date:
                y, mo, d = map(int, m_date.groups())
                current_date = datetime(y, mo, d)
                continue

            # 메시지 라인 (두 가지 형식 시도)
            m_msg = MESSAGE_RE.match(line) or MESSAGE_RE_ALT.match(line)
            if m_msg and current_date is not None:
                sender = m_msg["sender"].strip()
                h24, mm = _to_24h(m_msg["ampm"], int(m_msg["h"]), int(m_msg["m"]))
                ts = current_date.replace(hour=h24, minute=mm)
                msg = ParsedMessage(
                    relationship=relationship,
                    timestamp=ts,
                    sender=sender,
                    is_me=(sender == my_name),
                    text=m_msg["text"].strip(),
                )
                messages.append(msg)
                last_msg = msg
            else:
                # 매치 실패: 이전 메시지의 연속 라인으로 간주 (카톡은 줄바꿈을 이렇게 저장)
                if last_msg is not None:
                    last_msg.text += "\n" + line.strip()

    return messages


def parse_multiple(files: dict[str, Path], my_name: str = MY_NAME) -> pd.DataFrame:
    """여러 파일을 파싱해서 하나의 DataFrame으로 결합."""
    all_rows: list[dict] = []
    for relationship, path in files.items():
        msgs = parse_kakaotalk(path, relationship, my_name)
        all_rows.extend(m.to_dict() for m in msgs)
        print(f"  [{relationship}] {len(msgs):,}개 메시지 파싱")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["relationship", "timestamp"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="카카오톡 .txt 파일들을 DataFrame으로 파싱")
    parser.add_argument("inputs", nargs="+", help="입력 .txt 파일 경로들")
    parser.add_argument("--names", nargs="+", required=True, help="각 입력에 대응하는 관계 라벨")
    parser.add_argument("--my-name", default=MY_NAME, help=f"본인 닉네임 (기본값: {MY_NAME})")
    parser.add_argument("--output", default="data/messages.csv", help="출력 CSV 경로")
    args = parser.parse_args()

    if len(args.inputs) != len(args.names):
        parser.error("inputs와 names의 개수가 같아야 합니다")

    files = {name: Path(p) for name, p in zip(args.names, args.inputs)}
    for p in files.values():
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"파싱 시작 (본인 닉네임: {args.my_name})")
    df = parse_multiple(files, args.my_name)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n총 {len(df):,}개 메시지 → {out}")
    print(f"관계별 분포:\n{df['relationship'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
