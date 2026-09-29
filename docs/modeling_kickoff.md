# 모델링 착수 가이드 (팀 공유)

주제 ③ 프레스 유압펌프 이상 조기탐지. 이 문서의 PR이 병합되면 모델링(2x)·오류분석(3x) 노트북 작업을 시작한다.
근거: `reports/02_diagnosis_deep.md` §7·§9, `reports/03_diagnosis_recheck_CHS.md` §4.

## 1. 목적과 심사 2번 요구

- 심사 2번(40점): **베이스라인 포함 2개 이상 모델 비교**. 최종 산출은 `results/model_comparison.csv`(3x 분석 노트북이 각 `results/<노트북명>/metrics.csv`를 모아 생성, 직접 편집 금지).
- 모든 모델은 같은 입력(§2), 같은 전처리(§3), 같은 분할(§4), 같은 지표(§5)를 쓴다. 하나라도 다르면 비교표가 성립하지 않는다.

## 2. 입력 계약: `data/processed/11_window_features.parquet`

재생성: `notebooks/11_window_features_CHS.ipynb` 실행(`data/raw`가 없으면 zip에서 자동 추출). 윈도우 1행, `win_s`(1·2·3초)로 필터해 사용한다. 윈도우 수(정상/이상): 1초 3,264/96, 2초 2,254/63, 3초 1,400/38.

| 컬럼 | 의미 | 입력 가능 |
|---|---|---|
| `AI0_Vibration_*`, `AI1_Vibration_*`, `AI2_Current_*` `{rms, peak, p2p, kurt, crest, skew}` | 진폭 피처군(18열) | 가능 (기본) |
| `<채널>_shape_{kurt, crest, skew}` | 형상 피처군: 윈도우 내 z-score 후 값(9열). 진동 게인 동일성 미확인(03 A-2) 대비 | 가능 (ablation +형상) |
| `cur_fit_resid_rms`, `cur_fit_r2`, `cur_fit_f_dev` | 전류 사인 피팅 잔차 RMS · R² · \|f−0.6\|(3열). **세그먼트 전체 1회 피팅을 윈도우에 브로드캐스트한 세그먼트 수준 값** | 가능 (ablation +사인, 날짜 교란 가능성 주의) |
| `vib_corr01`, `cur_ac1` | 채널 관계(2열): 세그먼트 내 AI0·AI1 피어슨 상관, 전류 lag-1 자기상관. **세그먼트 수준 값 브로드캐스트**. reference EDA 지표(정상 +0.28 → 이상 −0.35, 0.93 → 0.44)와의 비교용이며 `cur_ac1`은 사인 잔차와 정보가 겹친다 | 가능 (ablation +관계) |
| `seg_uid` | 세그먼트 키(그룹) | 금지 (분할·집계 키로만) |
| `src` | 정상/이상 파일 | 금지 (날짜 = 라벨) |
| `label` | 정답(0 정상, 1 이상) | 금지 (정답) |
| `win_s`, `n_samples`, `t_start` | 윈도우 길이 · 샘플 수 · 세그먼트 내 시작 오프셋(s) | 금지 (`t_start`는 지연 계산용) |
| `t_abs` | 윈도우 시작 절대 시각 | **금지** (날짜 = 라벨, 시간 블록 정렬 전용) |
| `fold`, `block` | 분할 번호(§4). `block`은 정상만, 이상은 NA | 금지 |
| `state`, `vib_grade`, `cur_grade` | 운전 상태(정상만) · 채널별 soft label(이상만) | 금지 (사후 해석·FN/FP 분석용) |

입력 열 목록은 `features.feature_columns(df)`로 얻는다(32열). 판정 불가 세그먼트(길이 < 윈도우)는 윈도우가 없어 평가에서 빠진다: 이상 세그먼트 기준 1초 4/21(19.05%), 2초 8/21(38.10%), 3초 11/21(52.38%), 정상은 각각 11.52% / 24.54% / 39.90%. 성능 보고 시 이 비율을 함께 적는다.

## 3. 표준 전처리와 금지 목록

- 전처리는 `preprocess.preprocess(df)` 한 줄(형식 통일 + 세그먼트 평균 제거, 기본 인자). 11 노트북이 이미 적용한 결과가 parquet이므로 parquet을 쓰면 별도 전처리가 필요 없다.
- 새 피처를 만들 때도 원시 `data_quality.load()` 값이 아니라 `preprocess()` 결과에서 시작한다.
- **금지**: 원시값(형식 잔여 신호), 형식 피처(`preprocess.format_features`, 진단 전용), 절대 시간 관련 피처, 결함 주파수(FFT/스펙트럼) 피처(60 Hz가 0.6 Hz로 접힘). 사인 피팅 이탈은 별도 피처군으로만 허용.

## 4. 분할

- `fold`: 세그먼트 계층화 GroupKFold 5(seed=42), 전체 620 세그먼트 기준. 이상 세그먼트 fold별 5·4·4·4·4. 윈도우 길이(1·2·3초)와 무관하게 같은 열을 쓴다. **윈도우 표에 `split.group_kfold_seg`를 다시 호출하지 않는다**(윈도우가 없는 세그먼트가 빠져 fold가 달라짐).
- `block`: 정상 세그먼트 시간순 5블록(119~120개). forward-chaining으로 train = 블록 `< k`, test = 블록 `== k`(k=1..4), test에 `split.append_outliers`로 이상 윈도우 전체를 합친다.
- 두 분할 모두 `metrics.csv`의 `split` 열에 `group_kfold_seg` / `time_block`으로 보고한다. 마지막 블록(4)은 `state=1` 비율이 85.7%로 앞 블록(47~55%)과 달라 운전 상태 이동이 있다.

## 5. 지표

- `evaluate.to_metrics_row(model, split, fold, auc, fpr_sample, fpr_segment, delay_median_s)`가 `results/README.md` 공통 컬럼(`model, split, fold, auc, fpr_sample, fpr_segment, delay_median_s`)을 만든다. 전체 평균 행은 `fold="mean"`.
- 임계값은 **train fold 정상 점수의 q99**(`evaluate.threshold_from_normal`)만 쓴다. 이상 데이터로 임계를 고르지 않는다.
- `fpr_sample`은 정상 윈도우 초과 비율, `fpr_segment`는 초과 윈도우가 1개라도 있는 정상 세그먼트 비율. 둘 다 보고한다.
- 탐지 지연 `evaluate.detection_delay_s(..., win_s=윈도우 길이)`: 세그먼트별 첫 초과 윈도우의 **완성 시각**(t_start + win_s − 0.1 s, 02 §8 정의와 동일). 체감 지연 = 윈도우 내 지연 + burst 간격 중앙값 7.958초(`include_gap=True`). 실제 이상 발생 시점은 알 수 없으므로 7.958초는 구조적 하한이다. `delay_median_s`는 탐지된 세그먼트만의 중앙값이며, 미탐지 수(`n_detected / n_out_seg`)를 함께 적는다.
- 현장 활용(4번)용 연속 규칙: `evaluate.kofn_alarm(exceed, k, n)`.
- 참고 기준(규칙 베이스라인 smoke, 1초 AI0 rms, `results/11_window_features_CHS/metrics_smoke.csv`): group_kfold_seg 평균 AUC 0.7864 · fpr_sample 0.0109 · fpr_segment 0.0395 · delay 0.95초, time_block 평균 AUC 0.7786 · fpr_sample 0.0116 · fpr_segment 0.0384 · delay 0.90초. 모델은 이 값을 넘어야 의미가 있다.

## 6. soft label과 FP 후보

- FN 해석: 이상 세그먼트 3·19·20은 진동이 정상과 유사(`vib_grade`), 19·20은 전류는 확실한 이상(`cur_grade`). 진동 채널 모델의 미탐은 이 세그먼트에서 라벨 노이즈 가능성을 먼저 본다.
- FP 해석: 정상 저부하 구간(`state`)에서 오경보가 몰리는지 확인한다. 정상 의심 세그먼트 목록(04 `normal_suspects.csv`)은 있으면 FP 후보로 함께 본다.

## 7. 피처군 ablation 규칙

- 기본 = 진폭 18열. 추가 = +형상, +사인 잔차, +관계(`vib_corr01`·`cur_ac1`). reference식 비교로 "+전류 DC(세그먼트 평균, 형식 누수 의심 채널)"를 별도 행으로 넣어도 되나 표준 입력에는 포함하지 않는다.
- **최소 보고**: 기본, 기본+사인 두 행. 사인 잔차는 윈도우 단독 AUC가 1.0000(11 셀 5)이지만 세그먼트 수준 값이라 날짜 교란 가능성이 있으므로 기본 성능과 분리해 해석한다.
- 지도 상한 참고: 세그먼트 로지스틱 OOF AUC 진폭 0.997, 진폭+사인 0.9999(03 `leak_ablation.csv`).

## 8. 담당·노트북

| 노트북 | 내용 | 담당 |
|---|---|---|
| `21_model_rule_baseline_CHS` | 규칙 기반 이동 RMS 베이스라인 | CHS |
| `22_model_<name>_<INI-A>` | IsolationForest 또는 OC-SVM | `<INI-A>` |
| `23_model_<name>_<INI-B>` | AutoEncoder 또는 GMM/Mahalanobis | `<INI-B>` |
| `31_error_analysis_<INI>` | FN/FP 집중 조건, 비교표 `model_comparison.csv` | 미정 |

- 착수 조건: 이 문서의 PR 병합.
- 시작: `notebooks/_template.ipynb`를 복사해 `<번호>_<내용>_<이니셜>.ipynb`로 만들고 `NB`를 파일명과 같게 한다. 결과는 `RES / "metrics.csv"`에 필수(§5 컬럼). 경로는 `src/paths.py` 상수만, 그림·결과는 노트북 전용 폴더에만 쓴다.
- 자기 노트북이 아닌 것과 `src/` 공통 파일(`paths`·`preprocess`·`split`·`features`·`evaluate`) 변경은 작은 PR로 분리한다.

## 9. 일정

| 기간 | 작업 |
|---|---|
| 10-03 ~ 10-06 | 모델 노트북 21·22·23 |
| 10-06 ~ 10-07 | 31 오류분석, 비교표 `model_comparison.csv` |
| 10-07 ~ 10-08 | 보고서·발표자료 (제출 10-08 23:59) |
