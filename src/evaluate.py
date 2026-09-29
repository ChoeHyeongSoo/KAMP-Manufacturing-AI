"""주제 ③ 프레스 유압펌프 — 공통 평가 지표 (AUC · 오경보율 · 탐지 지연 · 임계값 · k-of-n).

모든 모델 노트북이 같은 정의로 `metrics.csv`(results/README.md 공통 컬럼)를 채우도록 고정한다.
점수는 "클수록 이상"이라고 가정한다. **임계값은 항상 train fold의 정상 점수 분위수**로만 정한다
(`threshold_from_normal`; 02 §7 `leakage_free_fpr_table`과 같은 원칙). 이상 데이터로 임계를 고르지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BURST_GAP_S = 7.958          # burst 간격 중앙값 — 체감 지연의 구조적 하한 (02 §7)
METRIC_COLS = ["model", "split", "fold", "auc", "fpr_sample", "fpr_segment", "delay_median_s"]


def auc(y, score) -> float:
    """ROC AUC. 클래스가 하나뿐이거나 점수가 비면 nan."""
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def fpr_sample(score_normal, thr: float) -> float:
    """정상 윈도우 중 점수가 thr을 초과하는 비율."""
    s = np.asarray(score_normal, dtype=float)
    return float((s > thr).mean()) if len(s) else float("nan")


def fpr_segment(score_normal, seg_uid_normal, thr: float) -> float:
    """초과 윈도우가 1개라도 있는 정상 세그먼트의 비율(윈도우가 있는 세그먼트만 분모)."""
    t = pd.DataFrame({"s": np.asarray(score_normal, dtype=float), "u": np.asarray(seg_uid_normal)})
    if t.empty:
        return float("nan")
    return float((t["s"] > thr).groupby(t["u"]).any().mean())


def detection_delay_s(score_outlier, seg_uid_outlier, t_start, thr: float, burst_gap_s: float = BURST_GAP_S,
                      include_gap: bool = False, delay_offset_s: float = 0.0) -> pd.DataFrame:
    """이상 세그먼트별 첫 초과 윈도우까지의 지연.

    반환 컬럼: seg_uid, detected, delay_in_window_s(= 첫 초과 t_start + delay_offset_s), delay_s.
    delay_offset_s는 "윈도우가 완성된 시점" 보정(02의 이동 RMS 지연은 마지막 샘플 기준이라 1초 윈도우면 0.9).
    include_gap=True면 delay_s = 윈도우 내 지연 + burst_gap_s(체감 지연), 아니면 delay_s = 윈도우 내 지연.
    미탐지 세그먼트는 detected=False, 지연 nan. 중앙값은 탐지된 세그먼트만으로 구한다.
    """
    t = pd.DataFrame({"s": np.asarray(score_outlier, dtype=float), "u": np.asarray(seg_uid_outlier),
                      "t": np.asarray(t_start, dtype=float)})
    rows = []
    for uid, g in t.groupby("u", sort=False):
        g = g.sort_values("t")
        hit = g[g["s"] > thr]
        if len(hit):
            d_in = float(hit["t"].iloc[0]) + delay_offset_s
            rows.append({"seg_uid": uid, "detected": True, "delay_in_window_s": d_in,
                         "delay_s": d_in + (burst_gap_s if include_gap else 0.0)})
        else:
            rows.append({"seg_uid": uid, "detected": False, "delay_in_window_s": np.nan, "delay_s": np.nan})
    return pd.DataFrame(rows, columns=["seg_uid", "detected", "delay_in_window_s", "delay_s"])


def threshold_from_normal(score_normal_train, q: float = 0.99) -> float:
    """train fold **정상** 점수의 q분위수. 임계값은 이 함수로만 만든다."""
    s = np.asarray(score_normal_train, dtype=float)
    s = s[~np.isnan(s)]
    return float(np.quantile(s, q))


def to_metrics_row(model: str, split: str, fold, auc, fpr_sample, fpr_segment, delay_median_s) -> dict:
    """results/README.md 공통 컬럼 순서(model, split, fold, auc, fpr_sample, fpr_segment, delay_median_s)의 dict."""
    return dict(zip(METRIC_COLS, [model, split, fold, auc, fpr_sample, fpr_segment, delay_median_s]))


def kofn_alarm(exceed: pd.Series, k: int, n: int) -> pd.Series:
    """최근 n개 윈도우 중 k개 이상 초과하면 경보(현장 운영용 연속 규칙). 한 세그먼트의 시간순 bool 시리즈를 넣는다."""
    return exceed.astype(int).rolling(n, min_periods=1).sum() >= k
