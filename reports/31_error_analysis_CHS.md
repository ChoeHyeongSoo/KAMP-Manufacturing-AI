# 주제 ③ 프레스 유압펌프 — 모델 비교표 및 FN·FP 오류분석 리포트 (31_error_analysis_CHS)

- 노트북: `notebooks/31_error_analysis_CHS.ipynb` (실행 결과 포함, 에러 셀 0)
- 목적: 모델 노트북(21·24·25·26·27)의 `metrics.csv`를 모아 비교표를 만들고, 미탐(FN)·오경보(FP)가 몰리는 조건과 그 원인(상관 구조 변화, 정상 부분공간 이탈, 분포 형태 차이)을 정리한다.
- 심사 기준 대응: 2번(모델 비교표), 3번(FN/FP 집중 조건), 4번(상태별 경보 운영 근거), 5번(라벨 노이즈·단변량 규칙 한계 관점의 FN 해석)
- 탐지 대상 표현: 유압펌프 모터-펌프 구동부 이상(센서 부착 부위를 확정할 수 없어 원인이 모터인지 펌프인지는 데이터로 구분하지 않는다).
- 근거 수준 표기: **[데이터]** 이 리포트의 CSV·노트북 출력으로 직접 확인 / **[추정]** 데이터에서 논리적으로 따라오나 직접 검증하지 않음 / **[가정]** 현장·운영 전제.
- 상태 번호: parquet·03·04 규약(**0 = 고부하, 1 = 저부하**). 21 노트북의 규칙 내부 번호(0 = 저부하)와 반대이므로 섞지 않는다.

| 절 | 핵심 결과 | 출처 |
|---|---|---|
| A 비교표 | 평균 행 142행(규칙 7종 + 모델 27종), 1초 gkf AUC 0.9998 이상 9개 모델, 규칙 베이스라인 0.7840~0.8696 | `results/model_comparison.csv` |
| B FN 이원화 | 규칙 7종(causal 변형 rule3e 포함)과 진동 2채널 AE는 3·19·20 미탐, 전류 입력 모델은 모두 탐지, copod·hbos 계열·deep_svdd만 19를 놓침 | `results/31_error_analysis_CHS/fn_by_model.csv`, `fn_category_summary.csv` |
| C FP 집중 | 27모델 중 19개가 고부하에 FP 집중, 윈도우 z-score 입력만 저부하로 반전, 04 의심 세그먼트와 겹침이 큼 | `fp_by_state.csv`, `fp_by_block.csv`, `fp_common_segments.csv` |
| D 상관 구조 | \|Δρ\|≥0.5인 쌍 93/465(31열, relwin 4열 포함), 핵심은 채널 관계 반전이며 causal 누적 상관에서도 재현 | `corr_break_normal_vs_outlier.csv` |
| E PCA | 이상 17세그먼트 전부 amp18에서도 SPE·T² 모두 정상 q99 초과 | `fn_pca_spe_t2.csv` |
| F KS | sine 3열 KS 0.99~1.00, KS와 \|Cliff δ\| 순위가 어긋나는 형태 차이 후보 4개 | `ks_segment_normal_vs_outlier.csv` |

원칙: 이상 이벤트는 1건(1초 기준 평가 가능 세그먼트 17개)이라 아래 FN/FP 결과는 모집단 추론이 아니라 **이 이벤트의 기술**이다. 표준화·PCA·임계는 train fold 정상으로만 적합했고, 검정 p값은 세그먼트 대표값에서만 계산했다.

---

## 1. A. 모델 비교표

**방법.** `results/*/metrics.csv`(21·24·25·26·27) 781행을 합친 뒤 fold 평균 행만 모아 `results/model_comparison.csv`(142행)를 만들었다. 분할×윈도우별 행 수는 group_kfold_seg(gkf)·time_block 모두 1초 34 / 2초 30 / 3초 7(3초는 규칙 7종만 있음). 21의 rule3e(rule3c의 causal 변형, `vib_corr01_cum`)가 새로 들어갔다. 27은 error_cases를 전 시드 저장하므로 B·C에서는 `seed` 열로 첫 시드만 쓴다. [데이터: 노트북 셀 5 출력, `model_comparison.csv`]

**발견.**
- gkf 1초 AUC / fpr_segment 상위: mcd_amp_sine 1.0000 / 0.0546, ocsvm_amp_sine 1.0000 / 0.0896, cnn_lstm_ae_cur1 1.0000 / 0.0666, cnn_lstm_ae_all3_res 1.0000 / 0.0979, cnn_deepant 0.9999 / 0.0317, gdn 0.9999 / 0.0426, mtad_gat 0.9999 / 0.0286, ocsvm_amp_rel 0.9999 / 0.1106, lstm_ad 0.9998 / 0.0349. [데이터: `model_comparison.csv`]
- 27 원시 신호 복원 계열: pca_raw_ae_all3 0.9995 / 0.0413, cnn_lstm_ae_all3_zwin 0.9994 / 0.1482, cnn_lstm_ae_all3 0.9927 / 0.0765, cnn_lstm_ae_vib2 0.9556 / 0.0750. 진동 2채널만 쓴 vib2가 가장 낮고 전류 1채널(cur1)이 1.0000이다. [데이터]
- 진폭 전용(접미사 없음) 분포 모델은 mcd_amp 0.9842 / 0.0451, ocsvm_amp 0.9951 / 0.0726, copod_amp 0.9751 / 0.0394, hbos_amp 0.9695 / 0.0547, deep_svdd 0.9359 / 0.0914. [데이터]
- 규칙 베이스라인(gkf 1초 AUC / fpr_segment): rule3 0.8696 / 0.0358, rule3c 0.8671 / 0.0303, rule3b 0.8058 / 0.0399, rule2 0.7996 / 0.0468, rule1 0.7864 / 0.0395, rule3d 0.7840 / 0.0419. time_block에서는 rule3c가 0.8841 / 0.0162로 규칙 중 최고다. [데이터]
- 세그먼트 오경보율 최저는 copod_amp_sine 0.0264, mtad_gat 0.0286, rule3c 0.0303, cnn_deepant 0.0317, 최고는 cnn_lstm_ae_all3_zwin 0.1482, ocsvm_amp_rel 0.1106. AUC가 거의 같아도 오경보율은 5배 가까이 벌어진다. [데이터]
- time_block 1초에서는 ocsvm_amp_rel fpr_segment가 0.2429로 gkf(0.1106)보다 크게 나빠진다. cnn_deepant는 AUC 0.9999 / 0.0387로 유지된다. [데이터]
- 3초 윈도우는 규칙 7종만 있다(gkf AUC rule3 0.9496 ~ rule3d 0.8932, rule3e 0.9272). [데이터]
- rule3c의 causal 변형 rule3e는 1초 gkf AUC 0.8435 / fpr_segment 0.0323, time_block 0.8841 → 0.8502 / 0.0162 → 0.0141로, 세그먼트 브로드캐스트 상관이 주던 순위 이득의 약 1/3이 미래 정보였음을 보인다(21 §7). [데이터]
- **`_sine`·`_rel` 접미사 모델은 입력에 세그먼트 단위 브로드캐스트 피처(전류 사인 적합, 방향·비율 피처)가 들어간다.** 03에서 사인 잔차 피처의 세그먼트 단위 AUC가 1.000으로 나왔고(`reports/03_diagnosis_recheck_CHS.md`), 이 값은 날짜 교란 가능성과 분리되지 않는다. 따라서 `_sine`·`_rel` 결과는 진폭 전용 결과(cnn_deepant·lstm_ad·mtad_gat·gdn 및 접미사 없는 분포 모델)와 분리해 읽는다. [추정: 교란 여부는 별도 ablation 필요]

**시사점.** AUC만으로는 상위 모델을 가르기 어렵다(상위 7개가 0.9998 이상). 순위는 fpr_segment와 시간 블록 분할의 안정성으로 가려야 한다. 규칙 → 비지도 모델의 AUC 차이(0.79~0.87 → 0.94~1.00)는 크지만 이상이 1건이라 이 차이의 신뢰구간은 아직 없다. [추정]

**활용 아이디어(2번).** 최종 비교표는 "진폭 전용"과 "세그먼트 피처 포함" 두 묶음으로 나눠 제시하고, 후자에는 날짜 교란 가능성을 한계로 명시한다. 그림: [AUC vs 세그먼트 오경보율](../figures/31_error_analysis_CHS/model_auc_vs_fpr_segment.png).

---

## 2. B. FN 이원화 — 모델별 미탐 이상 세그먼트

**방법.** 노트북별 `error_cases.csv`를 `fn_long`으로 통일하고, 미탐 세그먼트를 판정 불가(0·5·9·10: 1초 윈도우 없음, 평가 분모 밖) / 전 채널 조용(3·19·20) / 기타로 나눴다. [데이터: `fn_by_model.csv`, `fn_category_summary.csv`]

**발견.**
- 규칙 7종은 gkf 1초에서 3·19·20을 모두 놓친다(각 1 fold). 상태별 방향 기준화(rule3c·rule3d)와 그 causal 변형(rule3e)을 더해도 미탐 집합은 같다. [데이터: `fn_category_summary.csv`]
- 27의 진동 2채널 AE(cnn_lstm_ae_vib2)도 규칙과 똑같이 3·19·20을 놓친다(gkf 각 1 fold, time_block 3×3·19×4·20×4). 3채널 AE(cnn_lstm_ae_all3)는 gkf에서 20을 1 fold, time_block에서 3×1·19×3·20×3 놓친다. 전류를 입력에 넣은 변형(cur1·all3_res·all3_zwin·pca_raw_ae_all3)은 미탐 0이다. [데이터]
- 24~26의 21모델 중 gkf 1초에서 19를 놓친 것은 copod_amp·copod_amp_shape·copod_amp_rel·hbos_amp·hbos_amp_shape·deep_svdd 6종(각 1 fold)뿐이다. 나머지 15개(cnn_deepant·lstm_ad·mtad_gat·gdn·mcd 4종·ocsvm 4종·copod_amp_sine·hbos_amp_sine·hbos_amp_rel)는 미탐 0이다. [데이터]
- time_block 1초에서는 copod_amp_sine·copod_amp_rel·hbos_amp_shape·copod_amp·copod_amp_shape가 19를 4블록 모두, hbos_amp는 2블록 놓친다. copod_amp는 3도 1블록, copod_amp_shape는 4(기타)도 2블록 놓친다. 주력 모델(CNN·LSTM-AD·MTAD-GAT·GDN·MCD·OC-SVM)은 미탐 0이다. [데이터: `fn_category_summary.csv`]
- 범주별 미탐(모델×세그먼트 행 수): gkf 전 채널 조용 28, time_block 전 채널 조용 14 / 기타 1. [데이터]
- 판정 불가 0·5·9·10은 1초 윈도우가 없어 분모 밖이다(1초 기준 평가 가능 17세그먼트). [데이터]
- **점검(`fn_check_vs_metrics.csv`).** error_cases는 첫 시드만 저장돼 metrics(시드 평균)와 어긋난다. 21의 error_cases는 gkf 1초만 커버하므로 그 외 diff는 커버 범위 밖이다. 커버 범위 내 불일치 6건: gdn_w1s time_block(metrics 1.0 vs error_cases 0), deep_svdd_w1s gkf(0.667 vs 1), deep_svdd_w1s time_block(1.333 vs 0), deep_svdd_w2s time_block(0.333 vs 0), cnn_lstm_ae_all3_w1s gkf(1.333 vs 1), cnn_lstm_ae_all3_w2s time_block(1.0 vs 2). 27은 전 시드를 저장하므로 시드별 차이를 직접 확인할 수 있다. [데이터]

**시사점.** "3·19·20 미탐"은 **규칙 베이스라인과 진동 전용 입력의 한계**다. 진동 2채널만 쓴 복원 모델이 규칙과 같은 3개를 놓치고 전류 채널을 넣으면 잡히므로, 조용한 이상의 탐지 여부는 전류 AC 파형 정보가 가른다(19·20은 평균 제거 후 전류 AC RMS가 정상 중앙값보다 낮은 세그먼트인데도 파형 복원 오차로 구분된다). 주력 모델과 전류 입력 AE는 모두 탐지하므로 FN 서술은 입력 채널 기준으로 적는다. 모델 쪽 미탐은 copod·hbos·deep_svdd의 19에 한정된다. [데이터]

**활용 아이디어(3·5번).** FN을 (a) 단변량 규칙·진동 전용 입력의 한계(3·19·20)와 (b) 모델 특성에 따른 미탐(copod·hbos·deep_svdd의 19)으로 이원화해 보고서에 적는다. 현장 센서 구성에서 전류 채널은 선택이 아니라 필수로 둔다. 그림: [미탐 매트릭스](../figures/31_error_analysis_CHS/fn_matrix_gkf_1s.png).

---

## 3. C. FP 집중 조건 — 운전 상태·시간 블록·04 정상 의심 세그먼트

### C-1. 운전 상태

**방법.** gkf 1초 FP 행(888행, 27모델)을 정상 530세그먼트 분모로 환산해 상태별 FP율을 비교(Fisher 정확 검정, BH 보정). 분모 상태 비율은 고부하 42.64% / 저부하 57.36%. [데이터: 노트북 셀 12~13 출력, `fp_by_state.csv`]

**발견.**
- 27모델 중 **19개가 고부하에 FP 집중**(BH p<0.05), 7개는 차이 없음(mcd_amp_shape, ocsvm 4종, copod_amp_shape, cnn_lstm_ae_vib2), 1개(cnn_lstm_ae_all3_zwin)만 저부하 집중(고부하 3.54% vs 저부하 23.03%). [데이터: `fp_by_state.csv`]
- 예: cnn_deepant 고부하 7.52% vs 저부하 0.33%(OR 24.6), hbos_amp 12.39% vs 0.33%(OR 42.8), mcd_amp 9.29% vs 0.99%(OR 10.3), cnn_lstm_ae_all3_res 21.68% vs 0.99%(OR 27.8), pca_raw_ae_all3 7.52% vs 1.64%. [데이터]
- 윈도우 z-score 입력(all3_zwin)은 크기 정보를 지운 대신 저부하에서 오경보가 23%로 치솟는다. 크기를 지우면 오경보가 사라지는 것이 아니라 위치가 바뀐다. [데이터]
- OC-SVM 계열은 저부하 쪽이 오히려 높다(ocsvm_amp 6.19% vs 8.22%, ocsvm_amp_rel 9.73% vs 12.17%). 단 차이는 유의하지 않다. [데이터]

### C-2. 시간 블록과 반복 FP 세그먼트

**발견.**
- 대부분 모델은 블록 4(저부하 85.7% 블록)에서 FP율이 가장 낮다(cnn_deepant 0.93%, mtad_gat 0.93%). mcd_amp는 블록 2·3에 몰린다(10.1%·8.3%, 블록 0·1은 0.9%·1.0%). ocsvm은 블록과 무관하게 7~12%. [데이터: `fp_by_block.csv`]
- 공통 FP 세그먼트(27모델 중): normal_460 22개, normal_310 20개, normal_52 17개, normal_178 16개, normal_569 15개, normal_303·13 14개. 과반 FP 7개는 전부 고부하이고, 04 의심(`results/04_data_profile_CHS/normal_suspects.csv`)과 겹치는 것은 5개(460·52·178·569·303). [데이터: `fp_common_segments.csv`]
- 04 의심 세그먼트(26개, 분모의 4.9%)가 모델별 FP에서 차지하는 비율: cnn_lstm_ae_all3_zwin 1.3% ~ mtad_gat 68.8%. 예측·그래프 계열(cnn_deepant 44.4%, gdn 50.0%, lstm_ad 47.6%, mtad_gat 68.8%)이 높고 MCD·OC-SVM은 6.8~27%, 27 복원 계열은 8.6~27.3%. [데이터: 노트북 셀 14 출력]

### C-3. 규칙 rule1

**발견.** rule1 FP 세그먼트 21개는 전부 고부하이고 04 의심과 19개가 겹친다. fpr_segment 21/530 = 0.0396으로 21의 metrics 0.0395와 일치한다. 21의 error_cases에는 FP 목록이 없어 본 노트북이 재계산했다. [데이터: 노트북 셀 15 출력]

**시사점.** 예측·그래프 계열과 규칙의 오경보는 "고부하 정상 구간의 진동 상단 꼬리"에 몰린다. 이는 모델 결함이라기보다 정상 분포의 우측 꼬리 문제일 가능성이 크다. OC-SVM은 상태와 무관하게 7~12%를 내므로 경계 설정 방식이 다르다. [추정]

**활용 아이디어(4번).** 고부하 상태에서만 임계값을 올리거나 연속 k회 초과일 때만 경보를 내는 상태별 임계 운영의 근거로 쓴다. 04 의심 세그먼트를 정상 변동 사례로 정리해 경보 해석 가이드에 넣는다. 상태별 임계·AND 결합·k-of-n의 설계와 평가는 `docs/design_JIW.md` 2단계 경보 설계에서 진행한다(중복 회피). [가정: 현장에서 고부하 구간을 식별할 수 있음]

---

## 4. D. 정상 vs 이상 상관 구조 변화

**방법.** 1초 윈도우의 정상 전체 vs 이상 전체 Spearman 상관(중복 열 제거 후 31열, 11 relwin 4열 포함)의 쌍별 차이 ρ_이상 − ρ_정상. 윈도우가 비독립이라 p값은 보고하지 않는다. [데이터: `corr_break_normal_vs_outlier.csv`, 노트북 셀 17 출력]

**발견.**
- 465쌍 중 |Δρ|≥0.5가 **93쌍**(relwin 열이 낀 쌍 24). 상위 쌍은 전부 vib_rms_ratio·vib_corr01(_cum)·cur_fit_r2가 낀 쌍이다. 예: AI1_p2p–vib_rms_ratio −0.759 → +0.506(Δ 1.265), AI1_peak–vib_corr01_cum 0.684 → −0.458(Δ 1.142), AI2_rms–vib_corr01 0.682 → −0.369, cur_fit_r2–vib_corr01 0.709 → −0.307. [데이터]
- causal 누적 상관 `vib_corr01_cum`도 세그먼트 전체 상관과 같은 크기의 반전을 보여, 아래 "실시간 과대평가" 우려 중 상관 피처 자체에 대한 부분은 완화된다(규칙 성능의 과대평가 폭은 21 rule3e 참고). [데이터]
- 순수 진폭 쌍 중 큰 것은 AI1_p2p–AI1_crest −0.404 → +0.520(Δ 0.924). [데이터]

**시사점.** 정상에서는 채널 간 진폭·방향·적합도가 함께 움직이고 이상에서는 이 관계가 뒤집힌다. 다만 상위 쌍이 세그먼트 브로드캐스트·방향 피처를 포함하므로 실시간 윈도우 판정 근거로는 과대평가일 수 있다. [추정]

**활용 아이디어(5번).** 채널 관계(AI1 p2p–crest 부호 반전 등)를 피처로 직접 넣거나, 상관 붕괴를 이상 점수로 쓰는 그래프 계열(GDN·MTAD-GAT)의 해석 근거로 쓴다. 그림: [상관 변화 히트맵](../figures/31_error_analysis_CHS/corr_break_heatmap.png).

---

## 5. E. 정상 부분공간 이탈 — PCA SPE·T²

**방법.** 피처 집합 2개(진폭 18열 amp18, 전체 33열 all33)에 정상 train fold로 PCA(95% 분산 성분: amp18 9개, all33 13개, 모든 fold 동일)를 적합하고, 정상 OOF q99 대비 이상 세그먼트의 SPE·T² 비율을 계산했다. [데이터: `fn_pca_spe_t2.csv`, 노트북 셀 19 출력]

**발견.**
- **이상 17세그먼트 전부가 amp18에서도 SPE·T² 모두 정상 OOF q99를 초과한다.** [데이터]
- 조용한 3개의 amp18 spe_ratio / t2_ratio: 3번 4.29 / 3.73, 19번 3.41 / 3.08, 20번 3.68 / 3.89. 다른 이상 세그먼트는 spe 1.5~7.7, t2 1.9~70.5. [데이터]
- all33은 비율이 수천~수만 배(사인·관계 피처 효과)이며, 19번 t2 308이 이상 중 최저다. [데이터]

**시사점.** 이 결과는 사전 가설(착수 가이드 §6의 라벨 노이즈 해석: 조용한 이상 19는 정상 q99 안쪽일 것)과 **반대**다. 19·20은 단일 RMS 임계로는 못 잡지만 진폭 18열의 다변량 구조로는 정상 q99의 3~4배 벗어난다. 따라서 이들의 미탐은 "라벨 노이즈" 가설보다 "단변량 규칙의 한계"에 더 가깝다. 다만 이상 중에서는 가장 정상에 가까운 축에 속한다. [데이터 + 추정]

**활용 아이디어(3·5번).** 규칙 임계를 RMS 하나가 아니라 다변량 거리(T²·SPE)로 바꾸는 경량 대안을 현장 후보로 제시한다. 그림: [PCA 산점도](../figures/31_error_analysis_CHS/fn_pca_scatter.png).

---

## 6. F. 세그먼트 수준 KS 통계

**방법.** 세그먼트 대표값(중앙값) 정상 530 / 이상 17에 KS 검정(BH 보정)을 적용하고 12의 Cliff's δ와 순위를 비교했다. 위치 차이는 비슷하고 형태가 다른 피처를 찾는 것이 목적이다. [데이터: `ks_segment_normal_vs_outlier.csv`]

**발견.**
- 상위: cur_fit_r2·cur_fit_resid_rms KS 1.0000, cur_fit_f_dev 0.9925, cur_ac1 0.9491, AI2_crest 0.8509, AI2_kurt 0.7254, AI0_rms 0.7021. [데이터]
- 33열 중 BH p≥0.05는 7열: AI2_peak, AI1_p2p, AI1_crest, AI0_crest, AI0_shape_crest, AI0_kurt, AI0_shape_kurt. [데이터]
- KS 순위와 |Cliff δ| 순위가 크게 어긋나는 것: AI0_rms(KS 8위 vs δ 13위), AI0_peak(9 vs 12), vib_corr01(13 vs 6), AI2_p2p(24 vs 15) — 형태 차이 후보. [데이터]
- shape_kurt/skew는 12에서 확인된 진폭 kurt/skew와 완전 중복이라 δ가 NaN이다. [데이터]

**시사점.** 사인 적합 3종과 cur_ac1이 압도적이지만 날짜 교란 가능성이 있는 피처군이다. 진동의 AI0_rms·AI0_peak은 위치보다 형태 차이의 비중이 크다. [추정]

**활용 아이디어(5번).** 형태 차이 후보 4개를 분포 기반 모델(COPOD·HBOS)의 입력 우선순위 검토 대상으로 둔다.

---

## 7. 한계

- 이상 이벤트 1건(평가 가능 17세그먼트)이라 모든 결과는 이 이벤트의 기술이다. [데이터]
- error_cases는 첫 시드만 사용(24~26은 첫 시드만 저장, 27은 전 시드 저장이나 일관성을 위해 seed 0만 사용)해 metrics(시드 평균)와 6건 불일치(2절). 21은 FP 목록이 없어 C-3은 재계산했다. [데이터]
- 상관·PCA·KS는 윈도우 비독립이라 p값을 일반화할 수 없다(D는 p값 미보고). [데이터]
- 모델 간 쌍 차이 신뢰구간은 24~26에 윈도우 점수 파일이 없어 수행하지 못했다(27만 `scores.csv` 보유). [데이터]
- `_sine`·`_rel` 모델의 날짜 교란 가능성은 분리해 표기했을 뿐 해소하지 못했다. [추정]

## 8. 다음 단계

1. FN 서술 정정 docs PR은 반영 완료(`CLAUDE.md`·`docs/modeling_kickoff.md` §6). 추가로 "진동 전용 입력도 같은 3개를 놓친다"는 결과를 최종 보고서 FN 절에 반영.
2. JIW에게 24~26 error_cases 전 시드 저장 요청.
3. 21에 FP 목록 저장 보강.
4. 상태별 임계·AND 결합·k-of-n은 `docs/design_JIW.md` 2단계 경보 설계에서 진행(중복 회피).
5. 27 `scores.csv`(윈도우 점수)로 27 변형 간 부트스트랩 쌍 차이 CI 수행. 24~26도 점수 파일을 저장하면 모델 간 CI로 확장.
