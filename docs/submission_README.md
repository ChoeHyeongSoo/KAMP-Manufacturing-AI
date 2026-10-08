# 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석

제6회 K-인공지능 제조데이터 분석 경진대회, 주제 ③ 소스코드 제출물이다. 정상 데이터로 학습한 시계열 모델이 이상을 일찍 탐지하면서 오경보를 낮추는 2단계 경보 구조를 만들고, 오류 조건과 현장 활용 방안까지 분석했다.

## 1. 폴더 구조

```
.
├─ README.md                  # 이 파일
├─ requirements.txt           # Python 3.12 기준 고정 버전
├─ src/                       # 재사용 코드(.py). 노트북이 import. src/run_all.py가 전체 순차 실행
├─ notebooks/                 # 실행 결과를 포함한 노트북(번호순 산출물). _template.ipynb는 빈 틀
├─ data/
│  ├─ README.md               # 데이터 설명·입력 계약
│  ├─ raw/                    # 학습용 CSV 2개(press_data_normal.csv, outlier_data.csv)
│  └─ 3. 소성가공 예지보전 AI 데이터셋.zip   # 원본 압축 파일
├─ figures/<노트북명>/        # 노트북이 저장한 그림
├─ results/<노트북명>/        # 지표·오류 사례 CSV (results/README.md에 공통 컬럼 설명)
├─ reports/                   # 노트북별 요약 리포트(md)
├─ models/                    # 결정적 모델 가중치 일부(4절)
├─ predictions/               # 테스트 데이터 예측결과(6절)
└─ docs/reproducibility_check.md   # 전체 재실행 점검 기록
```

## 2. 환경 설치

- Python 3.12
- 의존 패키지: `pip install -r requirements.txt`
- torch는 CPU 빌드로 충분하다. `requirements.txt`의 `torch==2.14.0`이 설치되지 않으면 `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`로 설치한다.
- `requirements.txt` 마지막 묶음(transformers·accelerate·huggingface-hub·safetensors·tokenizers)은 경보 설명 챗봇(노트북 34_alarm_chatbot)용이며 선택 사항이다. 판정 성능과 무관하다.
- Windows에서 노트북 실행 시 한글 출력이 깨지지 않도록 `PYTHONUTF8=1` 환경변수를 설정한다.

## 3. 실행 방법

학습용 CSV 2개가 `data/raw/`에 이미 들어 있으므로 바로 실행할 수 있다.

```bash
# 저장소(압축 해제한 폴더) 루트에서 — 입력 계약 순서대로 노트북 전체를 실행(CPU 약 2.5시간)
python src/run_all.py
```

개별 노트북만 실행하려면 다음과 같이 한다.

```bash
cd notebooks
PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 23_model_ensemble_ridge_mcd_JSC.ipynb
```

- 노트북은 `notebooks/` 안에서 실행한다는 전제로 경로를 잡는다. 경로는 `src/paths.py` 상수만 사용한다.
- 실행 순서는 번호순이 아니라 입력 계약을 따른다. 실제 순서는 `src/run_all.py`의 `ORDER`를 따르며, 04는 11이 만드는 `data/processed/11_window_features.parquet`을 읽으므로 **11 다음**에 돌린다. 22 → 34(실시간 입력 검증) → 23 순서도 점수 파일 참조 때문에 지켜야 한다.
- `34_alarm_chatbot_JIW`는 로컬 LLM(약 3GB) 다운로드가 필요해 순차 실행에서 제외했다(`run_all.py`의 `SKIP`).
- `38_presentation_figures_CHS`는 발표용 그림 생성 노트북으로 순차 실행 대상이 아니다. 단독 실행하면 된다.
- 중간 산출물(`data/processed/`)은 노트북이 재생성하므로 제출물에 포함하지 않았다.

## 4. models/ 에 포함된 가중치

용량을 줄이기 위해 다음 4개 폴더만 포함했다. 모두 시드가 고정돼 노트북을 다시 실행하면 같은 결과가 나오는 노트북의 산출물이다.

| 폴더 | 내용 |
|---|---|
| `13_preprocess_dc_check_CHS` | 전처리(DC 제거) 방식 비교 모델 |
| `22_model_classical_timeseries_JSC` | 고전 시계열·트리 4종(Naive·Kalman·Ridge-VAR·Isolation Forest) |
| `23_model_ensemble_ridge_mcd_JSC` | 최종 Ridge-VAR + MCD |
| `34_realtime_input_validation_JSC` | 실시간 입력 검증용 Ridge-VAR |

torch 계열 모델(24~29: 예측·그래프·분포·복원 계열)의 가중치는 용량 때문에 포함하지 않았다. 해당 노트북이 가중치가 없으면 처음부터 재학습하고(시드 0~2) 저장한다. 재학습 결과는 부동소수 수준의 작은 차이만 있고 결론은 바뀌지 않는다(`docs/reproducibility_check.md` 4절).

## 5. 최종 모델 요약

최종 모델은 **Ridge-VAR + MCD의 포함 관계 2단계 경보**이다(노트북 23, 리포트 `reports/23_model_ensemble_ridge_mcd_JSC.md`). 센서 수치 형식을 통일하고 burst별로 1차 차분한 신호에서 다음 두 단계를 판정한다.

- **주의** = Ridge-VAR 1차 차분 잔차(`ridge_diff`)가 정상 학습 fold의 q99 임계값 초과. 민감한 점검 후보이다.
- **경보** = 주의 AND MCD(`diff_amp`, 진폭 피처의 마할라노비스 거리)도 초과. 두 관점이 동시에 넘을 때만 즉시 확인한다. 경보는 항상 주의의 부분집합이다.

time-block 평가에서 이상 세그먼트 17/17을 주의·경보 모두 탐지했고, 탐지 지연 중앙값은 0.9초이다. 세그먼트 오경보율은 주의 0.0170, 경보 0.0025이며 GroupKFold에서는 주의 0.0206, 경보 0이다. 경보에 AND를 쓰면 약한 합성 이상(강도 1)의 탐지율이 0.3505에서 0.1633으로 줄어드는 대가가 있어, AND로 Ridge를 교체하지 않고 두 단계로 나눠 운영한다. (수치 출처: `results/23_model_ensemble_ridge_mcd_JSC/summary.csv`, `operating_decision.csv`)

## 6. 테스트 데이터 예측결과

`predictions/` 폴더에 최종 모델의 예측결과가 있다. 열 상세와 판정 규칙은 `predictions/README.md`를 본다.

| 파일 | 내용 |
|---|---|
| `outlier_windows.csv` | 이상 파일 윈도우별 대표 예측 |
| `outlier_windows_detail.csv` | 윈도우별 전체 점수(split·fold·seed 포함) |
| `outlier_segments.csv` | 이상 세그먼트별 판정·첫 경보 시각 |
| `normal_holdout_segments.csv` | 정상 hold-out 세그먼트 경보(오경보 근거) |

생성 명령(학습 없이 저장된 결과를 읽어 만든다): `python src/export_predictions.py`

## 7. 노트북 번호 대역과 산출물 색인

번호 대역: **0x 진단 · 1x 전처리·피처 · 2x 모델 · 3x 분석**. 파일명은 `<번호>_<내용>_<표식>.ipynb`이다.

| 번호 | 내용 | 리포트 |
|---|---|---|
| 01~04 | 데이터 품질·추가 진단·재검증·프로파일(의심 세그먼트·드리프트) | `reports/01~04_*` |
| 11~13 | 윈도우 피처·피처 통계·전처리 DC 점검 | `reports/11~13_*` |
| 14 | burst 공백 보간 가능성 점검(보간하지 않는 근거) | `reports/14_*` |
| 21 | 규칙 기반 베이스라인 7종 | `reports/21_*` |
| 22 | 고전 시계열·트리 4종(Naive·Kalman·Ridge-VAR·Isolation Forest) | `reports/22_*` |
| 23 | Ridge-VAR(1차 차분) + MCD 결합, 1초 2단계 경보(최종 모델) | `reports/23_*` |
| 24~26 | 예측·그래프·분포 계열 모델 비교 | `reports/24~26_*` |
| 27 | 복원 계열(CNN+LSTM AE, 선형 PCA 대조군) | `reports/27_*` |
| 28 | 시계열 + 통계 결합, 2초 2단계 경보 | `reports/28_*` |
| 29 | 가이드북 LSTM-AE 재현(원본 방식 / 팀 계약 방식) | `reports/29_*` |
| 31 | 모델 비교표·FN/FP 오류분석·집계 규칙 민감도 | `reports/31_*` |
| 32 | 경보 근거 카드, 운영점(목표 오경보율 임계·2임계·시간당 환산) | `reports/32_*` |
| 33 | 합성 열화 조기경보 | `reports/33_*` |
| 34 | 경보 설명 챗봇(로컬 LLM, 순차 실행 제외) / Ridge-VAR 실시간 입력 검증 | `reports/34_*` |
| 35 | CNN+MCD 비교 체계 평가 상세(혼동행렬·ROC·임계 안정성) | `reports/35_*` |
| 36 | 분할 컨포멀 p-value: 정상 기준 보정과 타당성 점검 | `reports/36_*` |
| 37 | 최종 Ridge·MCD 2단계 경보 정적 대시보드 | `reports/37_*`, `figures/37_ridge_mcd_dashboard_JSC/dashboard.html` |
| 38 | 발표용 그림 생성 | `figures/38_presentation_figures_CHS/` |

## 8. 재현성 점검 결과

저장된 가중치와 점수 캐시를 모두 치운 상태에서 `src/run_all.py`로 노트북을 순서대로 다시 돌려 커밋된 결과와 비교했다(`docs/reproducibility_check.md`).

- 24개 노트북 전부 에러 셀 0, 미실행 셀 0, 합계 약 153분(CPU). 결정적 노트북(01~04, 11~14, 21, 22, 23, 32, 36 등)의 결과 CSV는 소요 시간 열을 제외하고 수치까지 동일했다.
- torch 재학습 노트북(24~29)은 부동소수 수준의 작은 차이가 있으나, 비교표 202행 중 182행이 동일하고 상위 모델 순위와 최종 결론(이상 17/17 탐지, 경보 오경보 감소)은 바뀌지 않았다.
