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
- 모델 비교표 `results/model_comparison.csv`는 분석 노트북(3x)이 각 `*/metrics.csv`를 모아 생성한다. 직접 편집하지 않는다.
