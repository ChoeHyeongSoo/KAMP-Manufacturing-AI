# results/ — 모델 평가 결과 (git 포함)

노트북을 다시 돌리지 않아도 팀원이 성능을 확인할 수 있도록 **표·수치(CSV)** 만 남긴다.
모델 가중치는 `models/`(git 제외)에 둔다.

## 규칙

- 노트북마다 `results/<노트북명>/` 하위 폴더에만 쓴다 (`FIG, RES = paths.nb_dirs(NB)`의 `RES`). 남의 폴더는 수정하지 않는다.
- 모델 노트북(2x)은 `RES / "metrics.csv"`에 아래 공통 컬럼으로 한 줄 이상 저장한다 — 비교표를 자동으로 합치기 위함.

  | 컬럼 | 예 |
  |---|---|
  | `model` | `rule_rms`, `iforest` |
  | `split` | `group_kfold_seg`, `time_block` |
  | `fold` | `0`~`4`, 전체 평균은 `mean` |
  | `auc`, `fpr_sample`, `fpr_segment`, `delay_median_s` | 지표값 (없으면 빈칸) |

- 필요하면 `cv_scores.csv`(fold별 상세), `error_cases.csv`(FN/FP 세그먼트 목록)를 같은 폴더에 추가한다.
- `error_cases.csv`는 FN·FP 모두, 시드가 여럿이면 **전 시드** 저장을 권장한다(첫 시드만 저장하면 metrics 시드 평균과 어긋난다, 31 점검).
- 모델 비교표 `results/model_comparison.csv`는 분석 노트북(3x)이 각 `*/metrics.csv`를 모아 생성한다. 직접 편집하지 않는다.
- `model_comparison.csv`의 세그먼트 F1 열(`tp, fp, fn, tn, n_normal, n_outlier, precision, recall, f1, fn_all, f1_all`, `f1_agg`, `f1_block_min`·`f1_block_max`)은 집계 규칙 any · 임계 train 정상 q 0.99 · skip_first 0 조건이다. `f1_agg`가 `gkf_pooled`이면 5 fold 합산, `block_mean`이면 블록별 계산 후 블록 평균(min·max는 블록 범위)이며, 분모 `n_normal`은 윈도우가 1개 이상인 정상 세그먼트 수다. 24~27은 시드 평균 fold 행에서 재구성해 tp·fp가 소수일 수 있다. 28 결합 행(모델명 `&`·`|`·`[k/n]`·`[상태별]`)은 28의 규칙 조건(2초, 결합 전 q99 고정) 그대로이며 k-of-n·상태별 행은 any·q 0.99 조건이 아니다. 28 `error_cases.csv`는 `rule` 열 형식이라 31 B·C에서 제외된다.
