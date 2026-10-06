# 실행 모델 이론·결과 정리 및 후속 시계열 모델 제안

- 작성일: 2026-10-04
- 대상: 프레스 유압펌프 모터-펌프 구동부 진동·전류 이상 조기탐지
- 선행 문서: `reports/00_data_input_suggestion_JSC.md`
- 목적: 현재까지 실행한 모델을 동일한 관점에서 정리하고, 아직 검증하지 않은 모델 중 현재 데이터에 실제로 의미가 있는 후보를 우선순위별로 제안한다.
- 범위: 모델 이론·실행 결과·후속 모델 설계까지 다룬다. 새 모델 구현과 학습은 다음 단계로 분리한다.

---

## 0. 결론 요약

현재 프로젝트는 이미 다음 다섯 계열을 검증했다.

1. 규칙 기반: RMS·운전 상태·상하부 진동 방향
2. 예측 기반: CNN DeepAnT·LSTM-AD
3. 그래프 기반: MTAD-GAT·GDN
4. 분포 기반: MCD·OCSVM·HBOS·COPOD·DeepSVDD
5. 복원 기반: CNN+LSTM Autoencoder·PCA reconstruction

따라서 앞으로는 모델 수를 무작정 늘리는 것보다 **현재 비교에 없는 이론적 축**을 채워야 한다.

### 후속 모델 우선순위

| 우선순위 | 후보 | 추가하는 이유 | 최종 권고 |
|---|---|---|---|
| 1 | **Ridge-VAR/선형 AR 예측** | CNN·LSTM 예측 성능이 비선형 구조 덕분인지 확인하는 선형 시계열 대조군 | 반드시 수행 |
| 2 | **Kalman filter/선형 상태공간 모델** | 실시간·causal 예측, innovation 기반 점수와 불확실성 제공 | 반드시 또는 Ridge-VAR와 함께 수행 |
| 3 | **Isolation Forest [amp18]** | 현재 분포 계열에 없는 트리 기반 비선형 기준선, 구현 비용이 매우 낮음 | 반드시 수행 |
| 4 | **소형 TCN forecasting** | LSTM 없이 causal dilated convolution으로 시간 구조를 학습 | 조건부 수행 |
| 5 | **Gaussian HMM** | 정상 고부하·저부하 상태와 상태 전이를 하나의 확률 모델로 표현 | 조건부 수행 |
| 6 | **USAD** | 두 Autoencoder의 adversarial 학습으로 복원 계열 확장 | 여유가 있을 때 |
| 운영 보강 | **Conformal threshold + EWMA/CUSUM** | 새 base model보다 오경보 제어·변화 누적에 직접 기여 | 실시간·앙상블 단계에서 우선 수행 |

Transformer·GAN·Diffusion·대형 foundation model은 현재 데이터에서 우선순위가 낮다. 입력이 1초 10샘플, 센서가 3개이고 실제 이상 이벤트가 1건뿐이어서 모델 용량의 이득을 검증하기 어렵다. 복원 계열에서도 15,915개 파라미터의 CNN+LSTM AE보다 선형 PCA가 더 안정적이었다.

### 다음 모델링 단계의 최소 조합

시간과 비교 가능성을 고려하면 다음 3개를 먼저 추가하는 것이 가장 효율적이다.

```text
Ridge-VAR forecasting  ─┐
Kalman innovation      ├─ 시계열·실시간 축
Isolation Forest amp18 ┘  분포 비선형 축
```

그다음 Ridge-VAR가 CNN보다 충분히 낮거나 시간 구조의 비선형성이 필요하다는 근거가 생겼을 때만 TCN을 추가한다.

---

## 1. 비교 전제와 데이터 제약

모델을 추천할 때 현재 데이터의 다음 제약을 우선한다.

| 제약 | 모델 선택에 미치는 영향 |
|---|---|
| 정상 20,000샘플, 이상 600샘플 | 대형 모델보다 작은 정상-only 모델 우선 |
| 이상 이벤트 1건 | 지도 분류와 이상 라벨 기반 튜닝 금지 |
| 센서 3개 | 큰 그래프·attention 모델의 추가 이득이 작을 가능성 |
| 10 Hz | 5 Hz를 넘는 실제 결함 주파수 모델링 불가 |
| 세그먼트 최대 50샘플 | 긴 문맥을 가정하는 Transformer·Matrix Profile에 불리 |
| 주 입력 1초 `(10, 3)` | 짧은 시퀀스에서 안정적인 선형·작은 CNN 모델 우선 |
| 정상·이상 날짜 분리 | 전류 기반 AUC 포화를 일반화 성능으로 해석 금지 |
| 고부하 정상에 FP 집중 | 상태·불확실성·임계값 모델링이 중요 |
| 1초 판정 불가 이상 4/21 | 성능과 함께 커버리지 필수 보고 |

모든 후속 모델은 `reports/00_data_input_suggestion_JSC.md`의 입력 계약을 따른다.

- 주 입력: 1초, 10샘플, 3채널, 이동 간격 0.5초
- 보조 입력: 1초 윈도우 진폭 피처 18개
- 세그먼트 경계 횡단·공백 보간 금지
- 같은 세그먼트의 윈도우는 같은 fold
- train 정상 점수 q99 임계
- `fpr_segment`, 탐지 수, 판정 가능률, time-block 결과 필수 보고

---

## 2. 전체 모델 지도

모든 모델은 “정상 패턴에서 얼마나 벗어났는가”를 계산하지만, 정상 패턴의 정의가 다르다.

```text
센서 시계열
   │
   ├─ 크기·방향 규칙 ───────────────▶ 규칙 점수
   │
   ├─ 다음 샘플 예측 ───────────────▶ 예측 오차
   │      ├─ CNN / LSTM
   │      └─ MTAD-GAT / GDN
   │
   ├─ 윈도우 전체 복원 ─────────────▶ 복원 오차
   │      ├─ PCA
   │      └─ CNN+LSTM AE
   │
   └─ 윈도우 피처의 정상 분포 ──────▶ 거리·밀도·경계 점수
          ├─ MCD / OCSVM
          ├─ HBOS / COPOD
          └─ DeepSVDD
```

### 계열별 대표 모델

| 계열 | 대표로 남길 모델 | 이유 |
|---|---|---|
| 규칙 | `rule3c_rms_state_dirz` | 해석 가능하며 최신 규칙 중 세그먼트 FP가 낮음 |
| 예측 | `cnn_deepant` | 가볍고 실제·합성 이상 균형이 가장 좋음 |
| 그래프 | `mtad_gat` | GDN보다 안정적이지만 CNN 대비 추가 이득은 작음 |
| 분포 | `mcd_amp` | 조용한 이상 탐지, 상태 간 FP 균형, 해석 가능 |
| 복원 | `pca_raw_ae` | CNN+LSTM AE보다 AUC·FP·연산량 모두 유리 |

최종 보고서에서 모든 변형을 같은 비중으로 소개할 필요는 없다. 위 대표 모델을 본문에 두고 나머지는 ablation 또는 실패·한계 사례로 정리하는 편이 명확하다.

---

## 3. 규칙 기반 모델

### 3.1 공통 원리

정상 train 데이터에서 통계량의 상위 분위수 또는 robust z-score 기준을 정하고, 이를 넘으면 이상으로 판단한다.

```text
score(x) > threshold_q99  →  anomaly
```

신경망이 없어 계산이 빠르고 경보 원인을 직접 설명할 수 있다. 반면 하나 또는 소수의 피처에 의존하므로 정상 분포의 다변량 구조와 복잡한 파형 변화를 놓친다.

### 3.2 실행 규칙

| 모델 | 핵심 아이디어 | 장점 | 한계 |
|---|---|---|---|
| `rule1_rms_ai0` | AI0 이동 RMS 상한 | 가장 단순한 기준선 | 작은 값·파형 이상·채널 관계 변화 미탐 |
| `rule2_rms_state` | 운전 상태별 RMS 임계 | 고·저부하 분리 | 상태 추정 오류 영향 |
| `rule3_rms_state_dir` | 크기 × 상하부 진동 방향 | 상관 반전 반영 | `vib_corr01`이 세그먼트 전체 값 |
| `rule3b_rms_state_ratio` | 크기 × 상하부 RMS 비율 | 상하부 상대 진폭 반영 | 개선이 제한적 |
| `rule3c_rms_state_dirz` | 운전 상태별 방향 robust z | 최신 규칙 중 오경보 개선 | 세그먼트 전체 상관으로 실시간 과대평가 가능 |
| `rule3d_rms_state_ratioz` | 운전 상태별 비율 robust z | 윈도우 단위 관계 사용 | rule1 수준으로 성능 하락 |

### 3.3 1초 결과

| 모델 | AUC (gkf / tb) | fpr_segment (gkf / tb) | 해석 |
|---|---|---|---|
| rule1 | 0.7864 / 0.7786 | 0.0395 / 0.0384 | 단변량 크기 기준선 |
| rule3 | 0.8696 / 0.8686 | 0.0358 / 0.0187 | 방향 결합으로 순위 개선 |
| **rule3c** | **0.8671 / 0.8841** | **0.0303 / 0.0162** | 최신 규칙 대표 |
| rule3d | 0.7840 / 0.7763 | 0.0419 / 0.0216 | 윈도우 RMS 비율만으로는 개선 없음 |

규칙 계열은 이상 3·19·20을 놓친다. 이 세 구간은 단일 RMS 상한에는 걸리지 않지만 다변량 PCA·MCD·예측 모델에서는 정상 구조를 벗어난다. 따라서 “라벨 오류”보다 **단변량 규칙의 한계**로 해석하는 것이 현재 결과에 맞다.

### 3.4 역할

- 모델 성능 비교의 최저 기준선
- 경보 사유 설명
- 딥러닝 모델의 추가 복잡도가 실제로 필요한지 판단
- 현장 fallback 규칙

실시간 모델로 발전시키려면 세그먼트 전체 상관을 사용하는 rule3·3c보다 윈도우 또는 과거 누적값만 사용하는 causal 규칙을 새로 만들어야 한다.

---

## 4. Forecasting 계열

### 4.1 공통 원리

정상 시계열의 과거 구간으로 다음 샘플을 예측한다. 정상 패턴이면 예측이 맞고, 이상이면 실제값과 예측값 차이가 커진다.

```text
과거 x[t-L:t] ─▶ model ─▶ x_hat[t]

score_t = mean_c ((x[t,c] - x_hat[t,c]) / sigma_error[c])²
```

복원 모델과 달리 미래 샘플을 입력으로 사용하지 않아 구조적으로 실시간 적용에 적합하다. 단 현재 공통 전처리의 세그먼트 전체 평균 제거는 별도의 causal 검증이 필요하다.

### 4.2 CNN DeepAnT

두 개의 1D convolution이 짧은 파형 패턴을 읽고 다음 3채널 값을 예측한다.

- 장점: LSTM보다 가볍고 병렬 계산 가능
- 강점: 진동 spike, 전류 파형 이탈
- 한계: 운전 상태가 바뀌면 정상 예측 오차 분포도 달라져 고부하 FP 발생
- 원 논문: 정상 시계열의 분포를 CNN 예측기로 학습하고 예측 오차로 이상을 찾는 방식

### 4.3 LSTM-AD

LSTM의 hidden state로 시간 순서를 압축한 뒤 다음 값을 예측한다.

- 장점: 순차 의존성을 명시적으로 모델링
- 강점: 합성 진폭 증가 유형에서 CNN보다 일부 우세
- 한계: 짧은 5~10샘플 문맥에서는 메모리 구조의 장점이 제한적

### 4.4 1초 결과

| 모델 | AUC (gkf / tb) | fpr_segment (gkf / tb) | time-block 탐지 | 합성 이상 평균 |
|---|---|---|---:|---:|
| **CNN DeepAnT** | **0.9999 / 0.9999** | 0.0317 / 0.0387 | 17/17 | 0.252 |
| LSTM-AD | 0.9998 / 0.9997 | 0.0349 / 0.0335 | 17/17 | 0.238 |

두 모델 모두 실제 이상을 잘 잡았지만, 전류 파형 이탈이 날짜·계측 경로 차이일 가능성을 배제할 수 없다. 예측 계열 대표는 속도와 합성 이상 탐지력을 고려해 CNN DeepAnT로 둔다.

---

## 5. Graph 계열

### 5.1 MTAD-GAT

MTAD-GAT는 센서 간 관계와 시점 간 관계를 attention으로 학습한다.

```text
feature attention + temporal attention + GRU
        ├─ next-step forecasting
        └─ window reconstruction
```

예측 오차와 복원 오차를 결합해 이상 점수를 만든다. 현재 구현은 원 논문의 확률적 복원부를 결정적 복원으로 단순화했다.

### 5.2 GDN

GDN은 각 센서를 그래프 노드로 놓고 센서 임베딩과 attention으로 정상 센서 관계를 학습한다. 정상 관계에서 벗어나 예측 오차가 커지는 센서를 이상 원인으로 볼 수 있다.

### 5.3 1초 결과

| 모델 | AUC (gkf / tb) | fpr_segment (gkf / tb) | time-block 탐지 | 합성 이상 평균 |
|---|---|---|---:|---:|
| **MTAD-GAT** | **0.9999 / 0.9997** | **0.0286 / 0.0394** | 17/17 | 0.258 |
| GDN | 0.9999 / 0.9951 | 0.0426 / 0.0458 | 16.75/17 | 0.199 |

상·하부 진동의 반대 흔들림을 모델링할 것으로 기대했지만 합성 관계 이상에서도 CNN보다 뚜렷한 이득이 없었다. 센서가 3개뿐이라 학습할 그래프가 작고 GDN의 top-k 구조가 사실상 완전 그래프가 되는 것이 원인 후보이다.

### 5.4 역할

- “채널 관계를 학습해도 추가 이득이 작았다”는 비교 결과
- 센서 수가 증가하는 향후 데이터의 확장 후보
- 현재 최종 운영 모델보다는 연구 비교군

---

## 6. Distribution·One-class 계열

이 계열은 원신호 대신 1초 윈도우의 진폭 피처 18개 또는 원신호 임베딩에서 정상 영역을 정의한다.

### 6.1 MCD

MCD는 정상 데이터의 중심과 공분산을 이상치에 강건하게 추정하고 Mahalanobis 거리를 점수로 사용한다.

```text
d²(f) = (f - mu)' Sigma^(-1) (f - mu)
```

피처가 각각 얼마나 큰지만 보는 것이 아니라, 정상에서 피처들이 함께 움직이는 방향을 본다. 그래서 고부하에서 진동과 전류가 함께 커지는 정상 패턴을 단순 상한 규칙보다 잘 수용한다.

### 6.2 One-Class SVM

RBF kernel 공간에서 정상 데이터를 감싸는 경계를 학습한다.

- 장점: 비선형 정상 경계
- 한계: 현재 결과에서 세그먼트 FP가 높고 time-block에서 불안정
- 정상에서 매우 먼 이상 점수가 포화되어 심각도 구분이 어려움

### 6.3 HBOS

각 피처의 정상 히스토그램 밀도를 독립적으로 추정하고 낮은 밀도의 값을 이상으로 본다.

```text
score(f) = sum_j -log p_j(f_j)
```

빠르고 해석이 쉽지만 피처 간 상관관계를 무시한다.

### 6.4 COPOD

각 피처의 경험적 누적분포에서 좌우 꼬리 확률을 구해 이상도를 계산한다. 별도 학습 파라미터가 적고 오경보가 낮지만, 여러 피처가 동시에 조금씩 이상한 조합을 약하게 평가할 수 있다.

### 6.5 DeepSVDD

신경망이 정상 윈도우를 잠재 공간의 중심 가까이 보내도록 학습한다.

```text
score(x) = ||phi(x) - center||²
```

현재 데이터에서는 반대 흔들림 합성 이상에 상대적으로 강했지만 전체 AUC와 FP 안정성이 낮았다.

### 6.6 1초 기본 입력 `amp18` 결과

| 모델 | AUC (gkf / tb) | fpr_segment (gkf / tb) | time-block 탐지 | 해석 |
|---|---|---|---:|---|
| **MCD** | 0.9842 / 0.9968 | 0.0451 / 0.0499 | 17/17 | 계열 대표, 양방향 다변량 거리 |
| OCSVM | 0.9951 / 0.9956 | 0.0726 / 0.1577 | 17/17 | time-block FP 큼 |
| HBOS | 0.9695 / 0.9717 | 0.0547 / 0.0911 | 16.5/17 | 피처 독립 가정 한계 |
| COPOD | 0.9751 / 0.9610 | **0.0394 / 0.0175** | 15.75/17 | FP는 낮지만 일부 미탐 |
| DeepSVDD | 0.9359 / 0.9424 | 0.0914 / 0.1066 | 16.67/17 | 복잡도 대비 불안정 |

`amp_sine` 결과는 실제 이상 AUC가 높고 합성 전류 이상 탐지가 개선되지만, 세그먼트 전체 사인 피팅값과 날짜 교란 가능성을 포함하므로 기본 결과와 분리한다.

---

## 7. Reconstruction 계열

### 7.1 공통 원리

정상 윈도우를 압축했다가 복원한다. 정상과 다른 파형은 잘 복원하지 못해 오차가 커진다.

```text
x_window ─▶ encoder ─▶ z ─▶ decoder ─▶ x_hat_window

score = mean((x_window - x_hat_window)²)
```

### 7.2 CNN+LSTM Autoencoder

Conv1d가 국소 패턴을 추출하고 LSTM이 시간 순서를 압축해 잠재 벡터 `z`를 만든다. decoder가 윈도우 전체를 복원한다.

- 파라미터: all3 기준 15,915개
- 학습: 정상 데이터, 200 epoch, seed 3개
- 장점: 채널별 복원 오차로 경보 기여 설명 가능
- 한계: 학습 비용이 크고 고부하 정상 복원 오차가 두꺼움

### 7.3 PCA raw-window reconstruction

1초 `(10,3)` 윈도우를 30차원 벡터로 펼치고 정상 데이터의 선형 부분공간을 학습한다. 부분공간에서 벗어난 복원 잔차를 이상 점수로 사용한다.

- 장점: 결정적, 빠름, 적은 파라미터, 해석 가능
- 한계: 비선형 시간 구조 표현 불가

### 7.4 1초 결과

| 모델 | AUC (gkf / tb) | fpr_segment (gkf / tb) | time-block 탐지 |
|---|---|---|---:|
| CNN+LSTM AE | 0.9927 / 0.9810 | 0.0765 / 0.0944 | 15.25/17 |
| **PCA raw AE** | **0.9995 / 0.9998** | **0.0413 / 0.0381** | **17/17** |

현재 데이터에서는 비선형 CNN+LSTM AE가 선형 PCA를 넘지 못했다. 따라서 새로운 복원 신경망을 추가하려면 “왜 PCA가 놓치는 이상을 새 모델이 잡을 것인가”라는 명확한 가설이 필요하다.

---

## 8. 현재 결과에서 얻은 모델 선택 원칙

### 8.1 AUC 포화

실제 이상에서 여러 모델의 AUC가 0.99~1.00이다. 이는 모델들이 모두 충분히 좋다는 뜻일 수도 있지만, 이상이 한 날짜·한 이벤트이고 전류 파형 차이가 매우 크다는 뜻이기도 하다.

앞으로 모델 순위는 다음 순서로 정한다.

1. `fpr_segment`
2. `time_block` 안정성
3. 평가 가능한 이상 세그먼트 수
4. 약한 합성 이상 탐지율
5. 고부하·저부하별 FP
6. causal 입력 여부
7. 계산량·재현성
8. ROC-AUC

### 8.2 복잡도가 성능을 보장하지 않는다

- PCA가 CNN+LSTM AE보다 안정적이다.
- CNN DeepAnT가 MTAD-GAT와 비슷하거나 더 효율적이다.
- GDN은 3채널 그래프에서 이득이 작다.
- DeepSVDD는 단순 MCD보다 불안정하다.

따라서 새 모델은 파라미터 수가 아니라 **현재 대표 모델과 다른 오차를 만드는가**로 선택한다.

### 8.3 최종 대표 조합

현재 결과만으로 최종 비교 후보를 줄이면 다음과 같다.

| 역할 | 모델 |
|---|---|
| 해석 가능한 기준선 | rule3c |
| 선형 원신호 복원 | PCA raw AE |
| 비선형 시계열 예측 | CNN DeepAnT |
| 피처 분포 | MCD amp18 |
| 채널 관계 비교군 | MTAD-GAT |

앙상블은 DeepAnT와 MCD처럼 서로 다른 계열을 우선한다. 예측·그래프 계열끼리는 FP 세그먼트가 50~60% 겹쳐 결합 이득이 작다.

---

## 9. 후속 추천 1 — Ridge-VAR/선형 AR 예측

### 9.1 원리

과거 여러 시점의 세 채널을 선형 결합해 다음 3채널 값을 예측한다.

```text
x[t] = A1 x[t-1] + A2 x[t-2] + ... + Ap x[t-p] + error[t]
```

Ridge 규제를 사용하면 짧고 상관된 입력에서도 계수를 안정적으로 추정할 수 있다.

### 9.2 왜 필요한가

현재 forecasting 계열에는 CNN과 LSTM만 있다. PCA가 복원 AE를 이긴 것처럼, 정상 전류의 0.6 Hz 파형과 세 채널 관계가 선형 예측만으로 충분할 수 있다.

Ridge-VAR를 넣으면 다음 질문에 답할 수 있다.

> CNN DeepAnT의 성능은 비선형 합성곱 덕분인가, 아니면 정상 파형이 단순해서 선형 예측만으로도 충분한가?

### 9.3 구현 제안

- 입력: 과거 5샘플 또는 10샘플 × 3채널
- 목표: 다음 1샘플 × 3채널
- 학습: train 정상 윈도우만 사용
- 규제: `Ridge(alpha)`; alpha는 정상 validation 예측 오차로만 선택
- 점수: 채널별 train 잔차 표준편차로 정규화한 제곱 예측 오차
- 분할·임계: 기존 24와 동일

### 9.4 기대 장점과 위험

| 장점 | 위험 |
|---|---|
| causal, 빠름, 계수 해석 가능 | 비선형 spike 대응이 약할 수 있음 |
| CPU에서 즉시 학습 가능 | 상태별 동역학 차이를 하나의 계수로 평균낼 수 있음 |
| CNN·LSTM의 필수 대조군 | 세그먼트가 짧아 높은 차수 p는 불안정 |

**추천 등급: 최우선.** 구현 비용 대비 비교 가치가 가장 크다.

---

## 10. 후속 추천 2 — Kalman filter/선형 상태공간 모델

### 10.1 원리

관측되지 않는 설비 상태 `z_t`가 시간에 따라 변하고 센서 `x_t`가 그 상태에서 생성된다고 가정한다.

```text
z_t = F z_(t-1) + process_noise
x_t = H z_t     + measurement_noise
```

Kalman filter는 이전 상태로 현재 센서값을 예측하고, 실제 관측과 예측의 차이인 innovation을 계산한다.

```text
innovation_t = x_t - x_hat_t
score_t = innovation_t' S_t^(-1) innovation_t
```

### 10.2 왜 필요한가

- 예측 오차뿐 아니라 오차의 기대 공분산을 함께 사용한다.
- 운전 상태가 안정적인 구간에서 작은 이탈을 누적해 볼 수 있다.
- 과거와 현재만 사용하는 완전한 causal 모델이다.
- 실시간 입력 검증 단계와 직접 연결된다.

### 10.3 구현 주의

- 각 세그먼트 시작에서 filter state를 초기화한다.
- 세그먼트 공백을 연속 시간으로 예측하지 않는다.
- 고부하·저부하를 하나의 모델로 처리할지 두 모델로 처리할지 ablation한다.
- 상태 차원은 작게 유지한다.

**추천 등급: 최우선 또는 Ridge-VAR와 한 노트북에서 비교.**

---

## 11. 후속 추천 3 — Isolation Forest [amp18]

### 11.1 원리

정상 피처를 무작위 축과 분할점으로 반복해서 나눈다. 다른 정상점과 빠르게 분리되는 관측일수록 짧은 트리 경로를 가지며 이상 점수가 높다.

### 11.2 왜 필요한가

현재 26번 분포 모델은 공분산, kernel 경계, 히스토그램, copula, 잠재 중심을 비교했지만 트리 기반 isolation 계열은 없다.

- 비선형 피처 상호작용을 볼 수 있다.
- scaling 민감도가 낮다.
- `scikit-learn`에 포함되어 추가 의존성이 없다.
- 학습과 추론이 빠르다.
- MCD의 타원형 정상 가정이 맞지 않을 때 보완할 수 있다.

### 11.3 구현 제안

- 입력: `amp18` 기본
- ablation: 비중복 amplitude+shape, causal 관계 피처
- 학습: train 정상
- 핵심 하이퍼파라미터: `n_estimators`, `max_samples`, `max_features`
- `contamination`으로 임계값을 자동 결정하지 않고 기존처럼 train 정상 score q99 사용

Isolation Forest 원 논문은 이상치를 고립시키는 random tree와 짧은 평균 경로 길이를 사용하며, 낮은 메모리와 빠른 계산을 장점으로 제시한다.

**추천 등급: 최우선.** 구현 비용이 가장 낮고 분포 계열 비교를 완성한다.

---

## 12. 후속 추천 4 — 소형 TCN forecasting

### 12.1 원리

TCN은 미래 값을 보지 않는 causal convolution과 dilation, residual block으로 시간 문맥을 모델링한다.

```text
causal dilated Conv1d → residual block → next-step prediction
```

원 TCN 비교 연구는 여러 sequence task에서 단순 convolution 구조가 표준 recurrent network보다 강한 출발점이 될 수 있음을 보였다.

### 12.2 현재 데이터에 맞춘 소형 구조

- 입력: 1초 `(10,3)`
- block 2개 이하
- hidden channel 8~16
- dilation 1, 2 정도
- kernel size 2 또는 3
- dropout 최소
- 목표: 다음 3채널 예측

### 12.3 기대 장점과 위험

| 장점 | 위험 |
|---|---|
| causal, LSTM보다 병렬화 쉬움 | 입력 10샘플이라 dilation 이득이 작을 수 있음 |
| spike·국소 패턴과 시간 문맥 결합 | DeepAnT의 일반 CNN과 차이가 작을 수 있음 |
| 미래 정보 없이 실시간 적용 가능 | 작은 데이터에서 구조만 복잡해질 수 있음 |

**추천 등급: 조건부.** Ridge-VAR가 DeepAnT보다 명확히 약하거나 DeepAnT가 놓치는 합성 유형이 확인될 때 수행한다.

---

## 13. 후속 추천 5 — Gaussian HMM

### 13.1 원리

관측되지 않는 운전 상태가 Markov transition에 따라 바뀌고, 각 상태에서 서로 다른 3채널 분포가 나온다고 가정한다.

```text
hidden state: high-load / low-load / transition
observation: vibration 2 + current 1
anomaly score: -log P(sequence | normal HMM)
```

### 13.2 왜 필요한가

현재 정상 데이터에는 고부하와 저부하 두 상태가 있다. 기존 모델은 `state`를 사후 분석 또는 임계 선택에 사용하지만 HMM은 상태와 상태 전이를 정상 모델 안에서 함께 표현할 수 있다.

### 13.3 주의점

- 각 세그먼트를 독립 sequence로 전달하고 경계에서 transition을 이어 붙이지 않는다.
- 상태 수는 정상 validation likelihood와 해석 가능성으로만 정한다.
- 현재 세그먼트가 짧아 상태 전이 확률이 불안정할 수 있다.
- `hmmlearn` 등 새 의존성이 필요할 수 있다.
- 상태 번호를 기존 0/1과 억지로 일치시키지 말고 사후에 대응시킨다.

**추천 등급: 조건부.** 상태별 FP 문제가 계속될 때 가치가 크다.

---

## 14. 후속 추천 6 — USAD

### 14.1 원리

USAD는 두 개의 Autoencoder를 adversarial 방식으로 학습해 정상 복원은 유지하면서 이상 입력의 복원 차이를 키우는 다변량 시계열 이상탐지 모델이다.

### 14.2 기대 효과

- CNN+LSTM AE와 다른 복원 학습 목적
- 정상-only 학습 가능
- 복원 오차 기반 채널 기여 가능
- 비교적 빠른 학습을 목표로 설계된 방법

### 14.3 현재 우선순위가 낮은 이유

- 기존 복원 실험에서 PCA가 CNN+LSTM AE보다 좋았다.
- 1초 입력은 30개 값뿐이라 복잡한 잠재 복원이 필요하지 않을 수 있다.
- adversarial 학습은 seed·학습률에 민감할 수 있다.
- 새로운 모델이 PCA보다 나아도 이상 이벤트 1건으로 일반화 근거를 만들기 어렵다.

**추천 등급: 여유가 있을 때.** 새로운 복원 계열이 반드시 하나 더 필요할 경우에만 수행한다.

---

## 15. 모델보다 먼저 또는 함께 검증할 운영 기법

### 15.1 Conformal threshold

현재 q99 임계는 단순하고 일관되지만 fold와 운전 상태에 따라 점수 분포가 달라질 수 있다. 정상 calibration block의 nonconformity score를 사용해 conformal p-value 또는 quantile threshold를 만들면 이상 라벨 없이 임계를 보정할 수 있다.

적용 대상:

- DeepAnT 예측 오차
- Ridge-VAR·Kalman innovation
- PCA reconstruction error
- MCD distance

최근 연구에서도 시계열 이상 임계 설정에서 calibration set만 사용해 누수를 막고 false alarm을 줄이는 conformal 접근이 제안되고 있다.

### 15.2 EWMA/CUSUM

한 번의 큰 score만 보는 대신 작은 score 상승이 지속되는지를 누적한다.

```text
EWMA_t = lambda * score_t + (1-lambda) * EWMA_(t-1)
```

또는 양의 편차를 누적하는 CUSUM을 사용한다.

- 장점: 약한 이상이 지속될 때 민감
- 단점: 세그먼트가 짧아 세그먼트 내부 누적 기회가 제한됨
- 처리: 세그먼트 경계에서 누적값 초기화 여부를 현장 사이클 정의에 맞춰 결정

### 15.3 상태별 calibration

고부하 정상에서 예측 계열 FP가 집중되므로, base model을 더 복잡하게 하기 전에 고부하·저부하별 정상 score 분포와 임계값을 분리하는 실험이 우선이다.

이 세 기법은 새 base model은 아니지만 실제 목표인 오경보 감소와 조기탐지에 직접 기여한다.

---

## 16. 우선순위가 낮은 모델과 이유

| 모델군 | 현재 우선순위가 낮은 이유 | 다시 검토할 조건 |
|---|---|---|
| Anomaly Transformer·TranAD | 긴 문맥과 많은 정상 시퀀스를 전제로 하는 attention 구조, 10샘플에서 과도함 | 더 긴 연속 데이터와 여러 날짜 확보 |
| PatchTST·TimesNet | 주로 긴 forecasting 문맥에서 강점, 현재 anomaly score 정의가 추가로 필요 | 장기 연속 시계열 확보 |
| GAN 계열 | 학습 불안정, 이상 1건에서 우열 검증 어려움 | 대규모 정상 데이터와 다수 seed 실험 가능 |
| Diffusion anomaly model | 계산량이 크고 30차원 입력에는 과도함 | 복잡한 고차원 센서 데이터 확보 |
| Matrix Profile | 긴 연속 시계열의 반복 motif·discord 탐색에 유리하지만 세그먼트 최대 50샘플 | 세그먼트를 잇는 물리적 근거가 생기고 장기 연속 데이터 확보 |
| ROCKET·MiniROCKET 분류 | 강한 지도 분류기지만 이상 이벤트 1건·날짜=라벨에서 과적합 위험 | 여러 날짜·여러 고장 이벤트 라벨 확보 |
| 더 큰 GNN | 센서가 3개라 graph capacity 이득이 작음 | 센서 수 증가 |
| 더 깊은 AE | 현재 PCA가 CNN+LSTM AE를 이김 | PCA가 놓치는 구체적 이상 유형 발견 |

Anomaly Transformer 자체는 association discrepancy를 이용하는 의미 있는 비지도 모델이지만, 이 프로젝트에서 사용하지 않는 이유는 모델의 수준이 낮아서가 아니라 **현재 입력 길이와 검증 데이터가 그 복잡도를 정당화하지 못하기 때문**이다.

---

## 17. 추천 실험 설계

### 17.1 1차: classical·tree 보강

권장 노트북:

```text
notebooks/22_model_classical_timeseries_JSC.ipynb
```

비교 모델:

1. naive persistence: `x_hat[t] = x[t-1]`
2. Ridge-VAR
3. Kalman innovation
4. Isolation Forest [amp18]

목적:

- 딥러닝이 단순 선형·naive 예측보다 실제로 나은지 확인
- MCD의 타원 가정과 Isolation Forest의 tree isolation 비교
- 완전 causal 모델의 기준 성능 확보

**실행 완료(2026-10-06):** [22 고전 시계열·트리 모델](22_model_classical_timeseries_JSC.md)에서 1초·9분할 전체 비교를 수행했다. Ridge-VAR는 time-block AUC 1.0000, `fpr_segment` 0.0145, 평가 가능 이상 17/17, 합성 이상 평균 0.2857로 DeepAnT(0.9999 / 0.0387 / 17/17 / 0.2538)를 앞섰다. Isolation Forest는 16/17·`fpr_segment` 0.0925로 MCD보다 불리했다. 사전 조건에 따라 **TCN은 보류**하며, Ridge-VAR의 causal 전처리 검증을 다음 단계로 둔다.

### 17.2 2차: 비선형 시계열 보강

권장 노트북:

```text
notebooks/23_model_tcn_usad_JSC.ipynb
```

진행 조건:

- 1차 모델의 결과로 비선형 시간 구조의 필요성이 확인될 것
- `causal_candidate_v1` 입력이 정해질 것

비교 후보:

1. small TCN forecasting
2. USAD optional
3. 기존 CNN DeepAnT·PCA와 직접 비교

### 17.3 3차: 임계·연속 경보

base model을 추가하는 대신 다음을 수행한다.

- normal calibration block conformal threshold
- 상태별 q99
- EWMA/CUSUM
- k-of-n
- OR=주의, AND=경보

---

## 18. 공통 성공 기준

새 모델은 AUC가 높다는 이유만으로 채택하지 않는다.

### 필수 비교 기준

| 기준 | 채택 조건 |
|---|---|
| 입력 | 기존 1초 계약과 동일 |
| 분할 | gkf 5 + time-block 4 |
| 임계 | train 정상 q99, 이상 라벨 사용 금지 |
| 커버리지 | 1초 평가 가능 17/21을 유지 |
| 실제 이상 | 탐지 수와 미탐 세그먼트 보고 |
| 정상 | `fpr_sample`, `fpr_segment`, 상태별 FP 보고 |
| 강건성 | seed·fold 변동, time-block 차이 보고 |
| 초기 이상 | 동일 합성 이상 4종·강도 4단계 사용 |
| 실시간성 | 미래 정보 사용 여부와 추론 시간 보고 |
| 설명성 | 채널별 오차·피처 기여 또는 상태 해석 제공 |

### 대표 모델 대비 질문

- Ridge-VAR·Kalman은 CNN DeepAnT에 비해 FP를 줄이는가?
- Isolation Forest는 MCD의 조용한 이상 탐지를 유지하면서 time-block FP를 줄이는가?
- TCN은 CNN DeepAnT가 약한 진폭 증가·반대 흔들림을 보완하는가?
- HMM은 고부하 FP를 줄이면서 저부하 이상 민감도를 유지하는가?
- USAD는 최소한 PCA reconstruction을 넘는가?

위 질문에 답하지 못하는 모델은 최종 후보에 추가하지 않는다.

---

## 19. 문서·결과 정합성 점검

현재 `results/model_comparison.csv`는 21·24·25·26 모델을 모은 시점의 결과다. 다음 항목은 아직 통합 비교표에 완전히 반영되지 않았다.

- 27 CNN+LSTM AE
- 27 PCA raw AE
- 최신 규칙 rule3c·rule3d
- 27의 윈도우 점수 기반 세그먼트 집계

따라서 새 모델을 추가하기 전에 비교표 생성 로직을 갱신해 기존 대표 모델과 새 모델이 한 표에 들어가도록 해야 한다. CSV를 수동 편집하지 않고 분석 노트북에서 다시 생성한다.

또한 `models/`의 학습 가중치는 git 제외 대상이므로, 결과 재현 시 노트북 실행 여부와 seed·hyperparameter·학습 시간을 함께 기록한다.

---

## 20. 최종 권고

### 모델링 단계에서 바로 수행

1. naive persistence
2. Ridge-VAR
3. Kalman innovation
4. Isolation Forest [amp18]

### 결과를 보고 조건부 수행

5. small TCN forecasting
6. Gaussian HMM
7. USAD

### 실시간·앙상블 단계에서 수행

8. 상태별 threshold
9. conformal calibration
10. EWMA/CUSUM 또는 k-of-n
11. DeepAnT OR/AND MCD 계열 결합

### 현재는 수행하지 않음

- 대형 Transformer
- GAN·Diffusion
- 더 깊은 graph model
- 지도 ROCKET 분류
- 긴 연속 신호를 전제한 Matrix Profile

최종적으로는 “가장 복잡한 모델”보다 다음 구조가 이 데이터에 적합하다.

```text
입력 1초 3채널
   ├─ causal forecasting: Ridge-VAR / Kalman / DeepAnT
   └─ feature distance: MCD / Isolation Forest
             ↓
      상태별·conformal 임계
             ↓
      연속 규칙과 2단계 경보
```

---

## 21. 참고 자료

### 저장소 내부

- `reports/00_data_input_suggestion_JSC.md`: 입력·세그먼트·분할 계약
- `reports/24_model_forecasting_JIW.md`: CNN DeepAnT·LSTM-AD
- `reports/25_model_graph_JIW.md`: MTAD-GAT·GDN
- `reports/26_model_distribution_JIW.md`: MCD·OCSVM·HBOS·COPOD·DeepSVDD
- `reports/27_model_cnn_lstm_ae_CHS.md`: CNN+LSTM AE·PCA reconstruction
- `reports/31_error_analysis_CHS.md`: 모델 비교와 FN·FP 분석
- `docs/model_guide_JIW.md`: 모델 이론 입문서
- `docs/design_JIW.md`: 2단계 경보 설계

### 원 논문·공식 자료

- DeepAnT: [DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series](https://www.dfki.de/en/web/research/projects-and-publications/publication/10175)
- MTAD-GAT: [Multivariate Time-Series Anomaly Detection via Graph Attention Network](https://doi.org/10.1109/ICDM50108.2020.00093)
- GDN: [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://ojs.aaai.org/index.php/AAAI/article/download/16523/16330)
- TCN: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- USAD: [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](https://www.kdd.org/kdd2020/accepted-papers/view/usad-unsupervised-anomaly-detection-on-multivariate-time-series.html)
- Isolation Forest: [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17)
- HMM anomaly detection: [Multivariate time series anomaly detection: A framework of Hidden Markov Models](https://doi.org/10.1016/j.asoc.2017.06.035)
- Matrix Profile: [Matrix Profile II](https://doi.org/10.1109/ICDM.2016.0085)
- Anomaly Transformer: [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://openreview.net/forum?id=LzQQ89U1qm_)
- Conformal threshold: [Conformalized Time Series Anomaly Thresholding with Latent Space Features](https://proceedings.mlr.press/v329/xu26a.html)
