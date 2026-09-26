# 주제 ③ 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석

데이터: `data/3. 소성가공 예지보전 AI 데이터셋.zip` → `data_raw/` (CSV 2개: normal 20,000행 / outlier 600행)

## 산출물
| 파일 | 내용 |
|---|---|
| `notebooks/01_data_quality.ipynb` | 데이터 품질 진단 (실행 결과 포함) |
| `reports/01_data_quality.md` | 진단 요약 리포트 — 이슈 Top 5, 필요 처리 표, 총평 |
| `src/data_quality.py` | 로딩·세그먼트·이동 RMS 함수 |
| `notebooks/02_diagnosis_deep.ipynb` | 추가 진단 — 세그먼트 등급·운전 상태·신호 해상도·에일리어싱·윈도우 민감도·탐지 지연 (실행 결과 포함) |
| `reports/02_diagnosis_deep.md` | 추가 진단 요약 리포트 — 절별 가설/결과/모델 단계 반영, 끝에 모델 단계 반영 사항 표 |
| `src/segments.py`, `src/signal_checks.py` | 세그먼트 피처·등급·운전상태·윈도우 민감도 / 신호 해상도·에일리어싱 진단 함수 |
| `figures/` | 리포트 그림 5장(01) + 7장(02) |

## 핵심 진단 (요약)
- 10 Hz, 최대 5초 burst 단위 수집. 정상 1일 77분 / 이상 **1일 2분 46초(이벤트 1건)**
- 날짜 = 라벨 → 절대 시간 피처 금지, 세그먼트 단위 분할 필수
- 전류는 60 Hz AC가 10 Hz로 에일리어싱된 파형 → FFT 무의미, 진폭 통계만 사용
- 이상 파일 안에 조용한 세그먼트 12%(FN 라벨 노이즈), 정상 안에 저부하 구간(FP 후보)
- (02) 이상 21세그먼트: 진동 기준 확실 이상 18/정상 유사 3(세그먼트 3·19·20), 전류 기준 확실 이상 7/경계 13/정상 유사 1 — 세그먼트 19·20은 진동은 조용하지만 전류 DC가 확실 이상 수준. 정상은 k-means로 고부하·저부하 2상태 분리
- (02) DC 오프셋 |값| 단독 AUC AI0 0.9443·전류 0.8911(1차 버전 계산 버그 수정 후 반전) → 파일 형식/양자화 차이(전류값 1.19209 정수배 비율 정상 0.19% vs 이상 99.67%)와 함께 강한 누수로 확인, **세그먼트 평균 제거 필수**
- (02) 윈도우 길이는 동일 세그먼트 집합 비교 기준 1~3초에서 AUC 개선(AI0 0.76→0.89), 오경보율은 샘플 단위 1.47%·세그먼트 단위 7.67%(정상 20% 학습 시)로 함께 보고, 탐지 지연 중앙값 0.9초(1초 윈도우 기준, burst 간격 포함 실제 체감 지연 ≈9.68초)

재실행: `cd notebooks && PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 01_data_quality.ipynb`
재실행(02): `cd notebooks && PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 02_diagnosis_deep.ipynb`
