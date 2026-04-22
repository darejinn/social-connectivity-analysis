# 사회적 연결성 Fermi 분석

건강과 사회 개인발표에 사용된 분석 코드입니다. 본인의 카카오톡 대화 로그를 직접 넣어 재현 가능한 분석을 돌려볼 수 있도록 정리했습니다.

> **데이터는 이 레포에 포함되어 있지 않습니다.** 사용자 본인의 카톡 로그를 직접 export 해서 넣어야 합니다.

---

## 빠른 시작 (본인 카톡으로 돌리기)

### 1. 설치

```bash
git clone https://github.com/darejinn/social-connectivity-analysis.git
cd social-connectivity-analysis
pip install pandas numpy matplotlib seaborn scipy
```

### 2. 카카오톡 대화 내보내기

**PC 버전 (Windows/Mac):**
1. 카카오톡 → 분석할 대화방 열기
2. 우측 상단 `≡` 메뉴 → `대화 내보내기` → `텍스트만 내보내기`
3. 저장된 `.txt` 파일을 `data/` 폴더에 넣기

**모바일:** 대화방 → 설정(≡) → `대화 내용 내보내기` → 이메일로 받아서 PC로 옮기기

예시 폴더 구성:
```
social-connectivity-analysis/
├── data/                      # ← 직접 만들어서 .txt 넣기 (gitignored)
│   ├── KakaoTalk_가족.txt
│   ├── KakaoTalk_친구A.txt
│   └── KakaoTalk_단톡B.txt
├── 01_parse_kakaotalk.py
├── ...
```

### 3. 본인 이름 설정

카톡 로그에서 "보낸 사람" 컬럼에 표시된 **본인의 닉네임**을 정확히 확인하세요 (공백·특수문자 포함).  
그 값을 아래 두 곳에 맞춰 넣습니다:

- `01_parse_kakaotalk.py`의 `MY_NAME` 상수
- `run_all.py` 실행 시 `--my-name` 인자

### 4. 전체 실행

```bash
python run_all.py \
  --inputs data/KakaoTalk_가족.txt data/KakaoTalk_친구A.txt data/KakaoTalk_단톡B.txt \
  --names 가족 친구A 단톡B \
  --my-name "본인닉네임" \
  --output-dir results/
```

결과물:
- `results/data/messages.csv`, `messages_labeled.csv`, `relationship_metrics.csv`, `sci_scores.csv`, `health_impact.csv`
- `results/figures/fig11_radar.png` ~ `fig16_fermi.png`


---

## 전체 파이프라인

```
[카카오톡 .txt 로그] ─► 01_parse_kakaotalk.py     (파싱)
                              │
                              ▼  messages.csv
                        02_level_classifier.py    (L1~L5 분류)
                              │
                              ▼  messages_labeled.csv
                        03_metrics_per_relationship.py   (관계별 지표)
                              │
                              ▼  relationship_metrics.csv
                        04_three_axis_scoring.py  (Holt-Lunstad 3축)
                              │
                              ▼  sci_scores.csv
                        05_fermi_health_impact.py (HR, 기대수명)
                              │
                              ▼  health_impact.csv
                        06_visualize.py           (그림 11~16)
```

## 파일 구성

| 파일 | 역할 | 입력 | 출력 |
|---|---|---|---|
| `01_parse_kakaotalk.py` | 카카오톡 txt 파싱 | `KakaoTalk_*.txt` | `messages.csv` |
| `02_level_classifier.py` | L1~L5 규칙 기반 분류 | `messages.csv` | `messages_labeled.csv` |
| `03_metrics_per_relationship.py` | 관계별 지표 계산 | `messages_labeled.csv` | `relationship_metrics.csv` |
| `04_three_axis_scoring.py` | Holt-Lunstad 3축 점수 | `relationship_metrics.csv` | `sci_scores.csv` |
| `05_fermi_health_impact.py` | HR·수명 Fermi 추정 | `sci_scores.csv` | `health_impact.csv` |
| `06_visualize.py` | 그림 11~16 생성 | 위 산출물들 | `figures/*.png` |
| `run_all.py` | 전체 파이프라인 일괄 실행 | — | 모든 산출물 |
| `test_with_mock_data.py` | 가상 데이터 검증 | — | 콘솔 출력 |

---

## 개별 단계 실행

```bash
# 파싱만
python 01_parse_kakaotalk.py data/KakaoTalk_가족.txt --names 가족 --my-name "본인닉네임" --output data/messages.csv

# Level 분류
python 02_level_classifier.py data/messages.csv --output data/messages_labeled.csv

# 시각화만 다시
python 06_visualize.py --output figures/
```

---

## 수정 포인트 (자주 변경되는 지점)

| 무엇을 바꾸고 싶을 때 | 파일 | 변수 / 함수 |
|---|---|---|
| 본인 이름 (카톡 닉네임) | `01_parse_kakaotalk.py` | `MY_NAME` |
| L1~L5 분류 기준 | `02_level_classifier.py` | `classify_level()` |
| 3축 가중치 | `04_three_axis_scoring.py` | `WEIGHTS` |
| Mehl 2010 건강 기준치 | `02_level_classifier.py` | `MEHL_HEALTHY_RATIO` |
| HR 값 (문헌 업데이트 시) | `05_fermi_health_impact.py` | `HR_TABLE` |
| 그림 색상 테마 | `06_visualize.py` | `COLORS` |
| 카톡 내보내기 형식이 다를 때 | `01_parse_kakaotalk.py` | `DATE_HEADER_RE`, `MESSAGE_RE` |

---

## 의존성

```
pandas  ≥ 2.0
numpy   ≥ 1.24
matplotlib ≥ 3.7
seaborn ≥ 0.12
scipy   ≥ 1.11
```



M
