# 주제 ① 사출성형 공정데이터 기반 품질불량 사전예측 및 검사 우선순위 결정

데이터: `data/1. 사출성형기 AI 데이터셋.zip` → `data_raw/` (CSV 4개: labeled/unlabeled × cn7/rg3)

## 산출물
| 파일 | 내용 |
|---|---|
| `notebooks/01_data_quality.ipynb` | 데이터 품질 진단 (실행 결과 포함) |
| `reports/01_data_quality.md` | 진단 요약 리포트 — 이슈 Top 5, 필요 처리 표, 총평 |
| `src/data_quality.py` | 로딩·품질 체크 함수 (`load_all`, `duplicate_report`, `idle_block_mask` 등) |
| `figures/` | 리포트 그림 6장 |
| `notebooks/02_diagnosis_deep.ipynb`, `reports/02_diagnosis_deep.md`, `src/pairs.py`/`drift.py`/`outliers.py` | 추가 진단 — 쌍 구조·타깃 정의·비가동 정밀 정의·시간 누수 정량화·cn7/rg3 결합 가능성·이상치 규칙 |

## 핵심 진단 (요약)
- 값이 파일별로 이미 z-score 표준화됨, 시간·설비 컬럼 없음
- labeled 전 행이 2회 중복(쌍), 불량 대부분이 "같은 X·다른 라벨" 충돌 쌍 → GroupKFold 필수
- unlabeled의 37~52%가 비가동 고정값 블록
- cn7에 시간 드리프트, 불량이 한 구간에 집중

재실행: `cd notebooks && PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 01_data_quality.ipynb`
