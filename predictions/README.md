# 테스트 데이터 예측결과 (predictions/)

최종 모델(노트북 23, Ridge-VAR + MCD 2단계 경보)의 저장 결과(`results/23_model_ensemble_ridge_mcd_JSC/`)를 읽어 만든 제출용 파일이다. 학습은 다시 하지 않는다.
생성 명령: `python src/export_predictions.py`

## 판정 규칙
- z = 점수 / 임계값. 임계값은 학습 fold의 정상 점수 q99이며 `z > 1`이 임계 초과이다(`src/ensemble_jsc.py` `run`).
- 주의(caution) = `z_ridge > 1` (ridge_diff), 경보(alarm) = `z_ridge > 1` 그리고 `z_mcd > 1` (ridge&mcd). 근거: `operating_decision.csv`.
- 세그먼트 판정은 세그먼트 내 1초 윈도우(0.5초 간격) 중 하나라도 초과하면 1(any).
- 윈도우 대표값은 group_kfold_seg의 fold x seed 평균 z이며, 평균 z에 같은 규칙을 적용한다.

## 파일
| 파일 | 행 수 | 내용 |
|---|---|---|
| `outlier_windows.csv` | 96 | 이상 파일 윈도우별 대표 예측. 열: seg_uid, t_start(세그먼트 내 윈도우 시작 초), state(0 고부하/1 저부하), z_ridge, z_mcd, caution, alarm, n_runs(평균에 쓴 fold x seed 수, 3) |
| `outlier_windows_detail.csv` | 1,440 | scores.csv의 label==1 전 행(split·fold·seed 포함) + caution·alarm. group_kfold_seg 288행, time_block 1,152행 |
| `outlier_segments.csv` | 21 | 이상 세그먼트 전부. 열: seg_uid, duration_s, n_windows, judgable, caution_any, alarm_any, first_caution_t_s, first_alarm_t_s(첫 초과 윈도우의 t_start), note |
| `normal_holdout_segments.csv` | 2,841 | 정상 하루 hold-out 세그먼트 경보(오경보 근거). 열: split, fold, seed, seg_uid, state, caution, alarm |

## 근거 수치
- 이상 세그먼트 21개 중 17개가 판정 가능하다. outlier_0(0.4초), outlier_5(0.3초), outlier_9(0.8초), outlier_10(0.4초)는 1초 윈도우를 만들 수 없어 판정 불가이다.
- 판정 가능 17개 모두 주의·경보가 발생했다(주의 17/17, 경보 17/17, 모두 첫 윈도우 t=0.0초에서 최초 초과). 윈도우 단위 경보율은 group_kfold_seg 0.948, time_block 0.965이다.
- 정상 세그먼트 오경보율(경보, ridge&mcd): time_block 3/1,251 = 0.0024(summary.csv fold 평균 0.00245), group_kfold_seg 0/1,590 = 0. 주의(ridge_diff)는 각각 0.0168, 0.0208이다.
- group_kfold_seg에서 각 이상 세그먼트는 한 fold의 test에만 들어가므로 윈도우당 n_runs는 3(seed)이다. time_block은 모든 fold에서 이상 세그먼트가 test에 포함된다.
- 이상 파일은 이벤트 1건(약 2분 46초)이다. 위 수치는 단일 이벤트 기준임에 유의한다.
