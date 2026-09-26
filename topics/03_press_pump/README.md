# 주제 ③ 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석

데이터: `data/3. 소성가공 예지보전 AI 데이터셋.zip` → `data_raw/` (CSV 2개: normal 20,000행 / outlier 600행)

## 산출물
| 파일 | 내용 |
|---|---|
| `notebooks/01_data_quality.ipynb` | 데이터 품질 진단 (실행 결과 포함) |
| `reports/01_data_quality.md` | 진단 요약 리포트 — 이슈 Top 5, 필요 처리 표, 총평 |
| `src/data_quality.py` | 로딩·세그먼트·이동 RMS 함수 |
| `figures/` | 리포트 그림 5장 |

## 핵심 진단 (요약)
- 10 Hz, 최대 5초 burst 단위 수집. 정상 1일 77분 / 이상 **1일 2분 46초(이벤트 1건)**
- 날짜 = 라벨 → 절대 시간 피처 금지, 세그먼트 단위 분할 필수
- 전류는 60 Hz AC가 10 Hz로 에일리어싱된 파형 → FFT 무의미, 진폭 통계만 사용
- 이상 파일 안에 조용한 세그먼트 12%(FN 라벨 노이즈), 정상 안에 저부하 구간(FP 후보)

재실행: `cd notebooks && PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 01_data_quality.ipynb`
