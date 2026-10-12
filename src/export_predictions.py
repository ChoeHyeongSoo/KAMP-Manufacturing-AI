"""테스트 데이터(이상 파일) 예측결과 내보내기.

학습을 다시 하지 않고 최종 모델(노트북 23, Ridge-VAR + MCD 2단계 경보)의 저장 결과만 읽어 제출용 CSV를 만든다.

입력 (results/23_model_ensemble_ridge_mcd_JSC/)
  - scores.csv          : 1초 윈도우(0.5초 간격) 점수. z_ridge, z_mcd = 점수 / 임계값(해당 fold 정상 train 점수의 q99).
                          따라서 z > 1이 임계 초과이다(ensemble_jsc.run).
  - segment_alarms.csv  : 정상 hold-out 세그먼트 경보(세그먼트 내 윈도우 중 하나라도 z > 1이면 True).
  - summary.csv         : 검증용 지표(fpr_segment, n_detected).
  - data/processed/segments.csv : 이상 세그먼트 21개의 길이(len, 10 Hz 샘플 수).

규칙
  - 주의(caution) = z_ridge > 1 (ridge_diff), 경보(alarm) = z_ridge > 1 and z_mcd > 1 (ridge&mcd).
  - 윈도우 대표값 = group_kfold_seg 의 fold x seed 평균 z. 평균 z 에 같은 규칙을 적용한다.
  - 세그먼트 판정 = 윈도우 중 하나라도 주의/경보이면 1(any). 1초 윈도우가 없는 세그먼트(len < 10)는 판정 불가.

출력 (predictions/)
  outlier_windows.csv, outlier_windows_detail.csv, outlier_segments.csv, normal_holdout_segments.csv

실행: python src/export_predictions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

SRC_DIR = paths.RESULTS / "23_model_ensemble_ridge_mcd_JSC"
PRED_DIR = paths.ROOT / "predictions"
FS = 10.0          # Hz
MIN_LEN = 10       # 1초 윈도우를 만들 수 있는 최소 샘플 수
REP_SPLIT = "group_kfold_seg"


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["caution"] = (df["z_ridge"] > 1).astype(int)
    df["alarm"] = ((df["z_ridge"] > 1) & (df["z_mcd"] > 1)).astype(int)
    return df


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(SRC_DIR / "scores.csv")
    seg_alarm = pd.read_csv(SRC_DIR / "segment_alarms.csv")
    summary = pd.read_csv(SRC_DIR / "summary.csv")
    segs = pd.read_csv(paths.DATA_PROCESSED / "segments.csv", encoding="utf-8-sig")

    out = scores[scores["label"] == 1]

    # 1) 상세: label == 1 전 행 + 플래그
    detail = add_flags(out)[["split", "fold", "seed", "seg_uid", "t_start", "state",
                             "z_ridge", "z_mcd", "caution", "alarm"]]
    detail.insert(4, "label", 1)
    detail.to_csv(PRED_DIR / "outlier_windows_detail.csv", index=False)

    # 2) 대표 윈도우 예측: group_kfold_seg fold x seed 평균
    rep = out[out["split"] == REP_SPLIT]
    g = rep.groupby(["seg_uid", "t_start"], sort=False)
    win = g.agg(state=("state", "first"), z_ridge=("z_ridge", "mean"),
                z_mcd=("z_mcd", "mean"), n_runs=("z_ridge", "size")).reset_index()
    win = add_flags(win)[["seg_uid", "t_start", "state", "z_ridge", "z_mcd", "caution", "alarm", "n_runs"]]
    win["_n"] = win["seg_uid"].str.extract(r"(\d+)$").astype(int)
    win = win.sort_values(["_n", "t_start"]).drop(columns="_n")
    win.to_csv(PRED_DIR / "outlier_windows.csv", index=False)

    # 3) 이상 세그먼트 21개
    osegs = segs[segs["label"] == 1][["seg_uid", "len"]].copy()
    osegs["_n"] = osegs["seg_uid"].str.extract(r"(\d+)$").astype(int)
    osegs = osegs.sort_values("_n")
    rows = []
    for _, r in osegs.iterrows():
        w = win[win["seg_uid"] == r["seg_uid"]]
        judgable = int(r["len"] >= MIN_LEN and len(w) > 0)
        row = {"seg_uid": r["seg_uid"], "duration_s": r["len"] / FS, "n_windows": len(w),
               "judgable": judgable}
        if judgable:
            c, a = w[w["caution"] == 1], w[w["alarm"] == 1]
            row.update(caution_any=int(len(c) > 0), alarm_any=int(len(a) > 0),
                       first_caution_t_s=c["t_start"].min() if len(c) else "",
                       first_alarm_t_s=a["t_start"].min() if len(a) else "",
                       note="")
        else:
            row.update(caution_any="", alarm_any="", first_caution_t_s="", first_alarm_t_s="",
                       note="1초 미만 세그먼트, 1초 윈도우 판정 불가")
        rows.append(row)
    seg_out = pd.DataFrame(rows)
    seg_out.to_csv(PRED_DIR / "outlier_segments.csv", index=False)

    # 4) 정상 hold-out 세그먼트 경보
    normal = seg_alarm.rename(columns={"ridge_diff": "caution", "ridge&mcd": "alarm"})
    normal = normal[["split", "fold", "seed", "seg_uid", "state", "caution", "alarm"]].copy()
    normal[["caution", "alarm"]] = normal[["caution", "alarm"]].astype(int)
    normal.to_csv(PRED_DIR / "normal_holdout_segments.csv", index=False)

    # 요약 + 검증
    j = seg_out[seg_out["judgable"] == 1]
    n_j = len(j)
    print("[행 수]")
    for f in ["outlier_windows", "outlier_windows_detail", "outlier_segments", "normal_holdout_segments"]:
        print(f"  {f}.csv: {len(pd.read_csv(PRED_DIR / (f + '.csv')))}")
    print(f"[이상 세그먼트] 판정 가능 {n_j}/21, 주의 {int(j['caution_any'].sum())}/{n_j}, "
          f"경보 {int(j['alarm_any'].sum())}/{n_j}")
    tb = detail[detail["split"] == "time_block"].groupby(["fold", "seed", "seg_uid"])["alarm"].max()
    print("[참고] time_block fold x seed별 경보 세그먼트 수:",
          sorted(set(tb.groupby(["fold", "seed"]).sum())), "(분모 = 이상 세그먼트가 test에 포함된 수)")
    for split, d in normal.groupby("split"):
        sm = summary[(summary["split"] == split) & (summary["model"] == "ridge&mcd")]["fpr_segment"].iloc[0]
        print(f"[정상 오경보율] {split}: 경보 {d['alarm'].mean():.5f} (summary {sm:.5f}), "
              f"주의 {d['caution'].mean():.5f}")


if __name__ == "__main__":
    main()
