# 재현성 점검 — 전체 노트북 순차 재실행 (2026-10-07)

> 심사 6번(코드 및 재현성, 10점) 근거. 평가 기준은 "전처리부터 결과 생성까지 자동 실행"이므로, 저장된 가중치(`models/`)와 점수 캐시(`data/processed/`)를 모두 치운 상태에서 `src/run_all.py`로 노트북 전체를 순서대로 다시 돌리고, 커밋된 결과 CSV와 비교했다.
> 수치의 근거 수준: **[데이터]** 이 점검의 로그·CSV로 확인 / **[추정]** 원인 해석.

## 1. 환경

| 항목 | 값 |
|---|---|
| OS / Python | Windows 11 / Python 3.12.10 (`.venv`) |
| `requirements.txt` 고정 버전 | 분석용 14종 전부 일치(pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.0, torch 2.14.0+cpu, pyod 3.6.6 등). 챗봇(34)용 5종(transformers 등)은 미설치 — 34는 이 점검에서 제외 |
| 가중치·캐시 | `models/`(13·27 가중치 11 MB)와 `data/processed/`(2.2 MB)를 저장소 밖으로 옮긴 뒤 시작. 끝난 뒤 `models/` 185 MB(9개 노트북), `data/processed/` 11 MB가 새로 생성됨 |
| 원본 데이터 | `data/raw/`는 그대로 두었다(`src/extract.py`는 노트북 첫 셀이 자동 호출) |

## 2. 실행 결과

`python src/run_all.py` (1차) 뒤, 1차에서 실패하거나 PR 대기 중인 노트북만 `--only`로 2차 실행. 합계 **153.0분**(CPU). [데이터: `run_all_log.csv` 1·2차]

| 순서 | 노트북 | 1차 | 소요 | 비고 |
|---|---|---|---|---|
| 1~3 | 01 · 02 · 03 | ok | 0.1 · 0.2 · 0.6분 | 02가 `segments.csv` 생성 |
| 4 | 04_data_profile_CHS | **error → 2차 ok** | 0.1분 | `11_window_features.parquet` 없음. **04는 11 다음에 돌려야 한다**(결함 ①) |
| 5 | 11_window_features_CHS | ok | 0.3분 | parquet 7,115행 × 49열, 커밋본과 동일 shape |
| 6 | 12_feature_statistics_CHS | **error → 2차 ok** | 0.3분 | `assert len(FEATS) == 33` 실패(37). `features.py`에 relwin 4열이 추가된 뒤 갱신되지 않은 assert(결함 ②, 이 PR에서 수정) |
| 7~9 | 13 · 14 · 21 | ok | 3.7 · 0.7 · 0.2분 | 13은 CNN 30 epoch 재학습 포함 |
| 10~13 | 24 · 25 · 26 · 27 | ok | 9.7 · 10.4 · 9.9 · **67.9분** | 전부 가중치 없이 재학습(시드 0~2) |
| 14~16 | 22 · 34_realtime_JSC · 23 | ok | 0.8 · 0.4 · 0.6분 | 22 → 34_realtime → 23 순서(cv_scores 참조) |
| 17 | 29_model_guidebook_lstm_ae_JIW | ok | 27.6분 | G2 재학습(시드 1개). G1은 `results/g1_runs.csv` 저장값 사용(CPU 약 85분/시드) |
| 18 | 28_model_ensemble_JIW | ok | 8.9분 | 점수 캐시 없이 24·26 가중치로 재수집 |
| 19 | 31_error_analysis_CHS | **error(main) → 2차 ok(#47)** | 2.5분 | main의 31은 23 `metrics.csv`에 `win_s` 등이 없어 `n_normal` assert 실패. PR #47 버전으로 재실행해 통과(결함 ③, #47 병합으로 해소) |
| 20~21 | 32_operating · 32_alarm_explain | ok | 2.3 · 1.6분 | 32_operating은 31 `f1_matched_fpr.csv`와의 정합 assert 통과 |
| 22~23 | 33 · 35 | ok | 0.4 · 0.2분 | 33은 캐시 없이 시나리오 재수집 |
| 24 | 36_conformal_pvalue_CHS | — → 2차 ok(#48) | 1.2분 | main에 아직 없음(PR #48). 재생성된 23 점수 위에서 실행 |
| 제외 | 34_alarm_chatbot_JIW | skipped | — | 로컬 LLM 약 3 GB 다운로드 필요. 판정 성능과 무관 |

2차까지 포함해 **24개 노트북 전부 에러 셀 0, 미실행 셀 0**. [데이터]

## 3. 발견한 결함과 조치

| # | 결함 | 영향 | 조치 |
|---|---|---|---|
| ① | README "시작하기" 순서(01~04 → 11~)대로 돌리면 04가 11의 parquet을 못 찾아 실패 | 순차 재현 실패 | `src/run_all.py` `ORDER`에서 04를 11 뒤로, README에 순서 주의 추가 |
| ② | 12의 `assert len(FEATS) == 33`이 `features.feature_columns` 확장(relwin 4열, 21 rule3d·3e용) 뒤 실패 | 12 실행 불가 | 12에서 `_win`·`_cum` 4열을 제외해 33열 유지(리포트 수치 불변, 12 `results/` CSV 동일) |
| ③ | main의 31이 23 `metrics.csv` 열 구성(`win_s`·`n_out_seg`·`n_detected` 없음) 때문에 실패 | 비교표 생성 불가 | PR #47에서 `cv_scores.csv`로 보완해 통과. #47 병합 필요 |
| ④ | `docs/project_guide_JIW.md` 383행에 개인 PC 경로(`C:\Users\…`)가 들어 있음 | 블라인드 평가 식별 정보 | 소유자(JIW) 정정 요청. 그 외 노트북·리포트·`src`·`docs`에서 소속·이름·도구명·절대경로 grep 결과 없음 |

## 4. 재실행 결과 대 커밋본 — 수치 차이

비교 기준: 31·`model_comparison.csv`는 PR #47, 36은 PR #48, 나머지는 main. 수치열은 절대차 1e-9 초과 셀 수, 소요 시간 열(`sec`)은 제외하지 않고 그대로 셌다. [데이터: `results_compare.csv`]

### 4.1 결정적 노트북 — 비트 수준 동일

| 노트북 | 결과 |
|---|---|
| 01~04, 11~14, 21 | `results/` 전부 동일(21 `state_fit_by_fold.csv` 최대 절대차 4e-14) |
| 22 고전 시계열(JSC) | `metrics.csv`·`scores.csv`(25,080행) 동일(최대 절대차 1e-13). `cv_scores.csv`·`summary.csv`는 **소요 시간 열(`sec`)만** 다름 |
| 23 Ridge+MCD(JSC) | `metrics.csv`·`cv_scores.csv`·`scores.csv` 동일. `timing.csv`만 다름 |
| 34_realtime(JSC) | `stream_timing.csv`(시간)와 그에 딸린 `summary.csv`·`cv_scores.csv` 시간 열만 다름 |
| 32_operating | `results/` 전부 동일(31 정합 assert 통과) |
| 36 conformal | CSV 9개 전부 동일(최대 절대차 2e-11, p-value·오경보율·calibration 크기 모두 일치) |
| 27 CNN+LSTM AE(CHS, torch) | `metrics.csv` mean 행 AUC·fpr_segment·n_detected 차이 **0**. `cv_scores.csv`는 `sec`만 다름 |

### 4.2 torch 재학습 노트북 — 작은 차이

시드(`torch.manual_seed`)는 고정돼 있지만 CPU 스레드 수·연산 순서가 달라 부동소수 결과가 완전히 같지는 않다. [추정] mean 행 기준 최대 절대차: [데이터]

| 노트북 | 모델 | Δauc | Δfpr_segment | Δn_detected |
|---|---|---|---|---|
| 24 | cnn_deepant 1초 / 2초 | 0 | 0.0013 / 0.0008 | 0 |
| 24 | lstm_ad | 0 | 0 | 0 |
| 25 | gdn 1초 | 0 | 0.0006 | 0 |
| 25 | mtad_gat | 0 | 0 | 0 |
| 26 | deep_svdd 1초 / 2초 | 0.0007 / 0.0001 | 0.0025 / 0.0009 | 0 |
| 26 | copod·hbos·mcd·ocsvm 전 변형 | 0 | 0 | 0 |
| 28 | cnn, cnn&mcd, [상태별] 변형 | 0 | ≤ 0.0010 | 0 |
| 28 | lstm·gat·mcd와 그 결합 | 0 | 0 | 0 |
| 29 | lstm_ae_guidebook 1초 / 2초 | 0.0080 / **0.0392** | 0.0076 / 0.0092 | 0 / 1.0 |

- 29(가이드북 LSTM-AE)만 차이가 크다. 시드 1개·단일 모델이라 평균으로 상쇄되지 않고, 2초 gkf AUC 0.9218 → 0.8826, time_block n_detected 10.25 → 9.25다. 29 리포트의 "팀 계약 아래에서는 예측 계열이 더 낫다"는 결론 방향은 바뀌지 않는다. [데이터]
- 24·26의 fpr_segment 차이는 세그먼트 1개(fold당 약 1%p)의 1/3 이하(3시드 평균)다. [데이터]

### 4.3 결론에 미치는 영향 — 없음

| 확인 항목 | 커밋본 | 재실행 | 판정 |
|---|---|---|---|
| `model_comparison.csv` 202행 중 수치가 바뀐 행 | — | 20행(cnn·gdn·deep_svdd·lstm_ae_guidebook·28 cnn 계열) | 182행 동일 |
| 세그먼트 F1 상위 5개(1초 gkf / 1초 time_block / 2초 gkf / 2초 time_block) | ridge&mcd, ridge_diff, copod_amp_sine, ridge_var, mtad_gat / … | **4개 그룹 모두 순위 동일** | 유지 |
| 28 vs 23 공통 세그먼트(31 A-5): 28 gkf 주의 fp | 12.67 / 452 (0.0280) | 12.33 / 452 (0.0273) | 유지 |
| 28 gkf 경보 fp | 1.00 / 452 (0.0022) | 0.67 / 452 (0.0015) | 유지 |
| 23 행 8개 | — | 전부 동일 | 유지 |
| 오경보율 차이(23 − 28) 95% 구간 4개 | 모두 0 포함 | 모두 0 포함(gkf 주의 [−0.0147, 0.0029], 경보 [−0.0037, 0.0000]) | 유지 |
| 이상 13/13 탐지(두 체계·두 단계) | 13/13 | 13/13 | 유지 |
| 31 FN 표(`fn_by_model.csv`) | 174행 | 175행(29 2초 gkf에 outlier_17 1건 추가, 29 재학습 차이) | 결론(규칙 3·19·20 / 모델 19) 유지 |

## 5. 재현 절차 (보고서 6절용)

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt                     # torch는 CPU 빌드
# data/ 에 원본 zip을 둔 뒤
python src/run_all.py                               # 노트북 24개 순차 실행, 로그 run_all_log.csv (CPU 약 2.5시간)
python src/run_all.py --only 24_model_forecasting_JIW,28_model_ensemble_JIW   # 일부만
```

- 가중치·캐시가 없으면 24~27·29(G2)·13은 학습부터 다시 하고, 28·33은 점수를 다시 수집한다. 있으면 불러온다(`RETRAIN`·`RECOLLECT` 플래그).
- 결정적 결과(규칙·고전·MCD·PCA·conformal)는 비트 수준으로 재현되고, torch 학습 모델은 fpr_segment 기준 0.003 이내(29만 0.01·AUC 0.04 이내)의 차이가 난다. 결론·순위는 바뀌지 않는다.
- 비교: `python src/repro_compare.py`가 재실행 결과를 커밋본과 대조해 파일별 shape·최대 절대차·초과 셀 수를 출력한다.

## 6. 한계

- 한 환경(Windows, CPU)에서 1회 재실행한 결과다. macOS·Linux나 다른 CPU 스레드 수에서는 torch 모델 차이 폭이 다를 수 있다. [추정]
- 34 챗봇은 제외했고, 29 G1(가이드북 원본 방식)은 저장된 실행 결과를 읽는다.
- 재실행으로 바뀐 노트북 출력·결과 CSV·그림은 커밋하지 않았다(커밋본은 각 소유자의 실행 결과). 이 문서와 `src/run_all.py`, 12 수정, README 순서 주의만 반영했다.
