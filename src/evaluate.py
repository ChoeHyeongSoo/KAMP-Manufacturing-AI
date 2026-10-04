"""주제 ③ 프레스 유압펌프 — 공통 평가 지표 (AUC · 오경보율 · 탐지 지연 · 임계값 · k-of-n · 세그먼트 F1).

모든 모델 노트북이 같은 정의로 `metrics.csv`(results/README.md 공통 컬럼)를 채우도록 고정한다.
점수는 "클수록 이상"이라고 가정한다. **임계값은 항상 train fold의 정상 점수 분위수**로만 정한다
(`threshold_from_normal`; 02 §7 `leakage_free_fpr_table`과 같은 원칙). 이상 데이터로 임계를 고르지 않는다.

세그먼트 F1: 평가 목표 지표는 세그먼트 단위 F1이다. 윈도우 F1은 이상 96 vs 정상 3,264 윈도우의 불균형으로
임계에 과민하고 현장 의미가 없다. 경보는 세그먼트(burst) 단위로 나가므로 F1도 세그먼트 단위로 정의한다
(`segment_alarms` → `segment_confusion` → `f1_vs_threshold`). 1초 윈도우 기준 이상 세그먼트 17개(판정 가능)·
정상 530개라 주력 모델이 17/17을 잡으면 F1은 FP 수가 결정한다(gkf 5 fold를 합산한 기준으로 fpr_segment 4% ≈ FP 21개 →
F1 ≈ 0.62, FP 4개 → 0.89; fold별 F1은 양성이 2~5개뿐이라 분산이 크므로 `pool_confusion`으로 합산해 보고한다.
time_block은 이상 전체가 블록마다 반복되므로 합산하지 않고 블록별로 본다). 그래서 FP를 줄이는 지렛대(세그먼트 집계 규칙 any/kofn/quantile, 시작 윈도우 제외, 임계 수준)를
같은 함수로 비교할 수 있게 한다. 판정 불가 세그먼트(1초 윈도우가 없는 길이 < 10 세그먼트: 이상 4/21, 정상 69/599)는
분모 선택지로 둔다 — `assessable`(윈도우 있는 세그먼트만, 기존 fpr_segment와 같은 분모) / `all`(해당 split의 이상
세그먼트 수를 주면 판정 불가를 FN으로 센다). 집계 규칙(`skip_first`·`kofn`)이 판정 대상 집합을 바꾸지 않도록 분모는
항상 "윈도우가 1개 이상인 세그먼트"로 고정하고, 규칙 때문에 판정할 윈도우가 없어진 세그먼트는 경보 없음(정상 TN·이상 FN)으로
센다. 이 모듈의 F1 곡선은 임계 수준별 F1을 "보고"할 뿐 F1이 최대인 임계를 고르지 않는다(이상 1이벤트에 과적합된다).
운영점은 정상 데이터만으로 정의한 허용 세그먼트 오경보율로 정한다(`threshold_for_segment_fpr`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BURST_GAP_S = 7.958          # burst 간격 중앙값 — 체감 지연의 구조적 하한 (02 §7)
METRIC_COLS = ["model", "split", "fold", "auc", "fpr_sample", "fpr_segment", "delay_median_s"]
F1_COLS = ["tp", "fp", "fn", "tn", "n_normal", "n_outlier", "precision", "recall", "f1", "fn_all", "f1_all"]   # 31 비교표의 세그먼트 F1 열 이름(조건 열 q·rule·k·n·skip_first·win_s는 31이 함께 저장)


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


def detection_delay_s(score_outlier, seg_uid_outlier, t_start, thr: float, win_s: float = 0.0,
                      fs: float = 10.0, burst_gap_s: float = BURST_GAP_S,
                      include_gap: bool = False) -> pd.DataFrame:
    """이상 세그먼트별 첫 초과 윈도우까지의 지연.

    반환 컬럼: seg_uid, detected, delay_in_window_s, delay_s.
    - delay_in_window_s = 첫 초과 윈도우의 **마지막 샘플 시각** = t_start + win_s − 1/fs. 윈도우가 채워져야
      판정이 나오므로 시작 시각이 아니라 완성 시각을 지연으로 센다(02 §8의 이동 RMS 지연과 같은 정의:
      1초 윈도우가 세그먼트 첫 샘플부터 초과하면 0.9초). win_s=0이면 t_start 그대로.
    - include_gap=True면 delay_s = delay_in_window_s + burst_gap_s(체감 지연), 아니면 delay_in_window_s.
    미탐지 세그먼트는 detected=False, 지연 nan. 중앙값은 탐지된 세그먼트만으로 구한다.
    """
    t = pd.DataFrame({"s": np.asarray(score_outlier, dtype=float), "u": np.asarray(seg_uid_outlier),
                      "t": np.asarray(t_start, dtype=float)})
    offset = max(win_s - 1.0 / fs, 0.0) if win_s > 0 else 0.0
    rows = []
    for uid, g in t.groupby("u", sort=False):
        g = g.sort_values("t")
        hit = g[g["s"] > thr]
        if len(hit):
            d_in = float(hit["t"].iloc[0]) + offset
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


def segment_alarms(score, seg_uid, t_start, thr: float, rule: str = "any", k: int = 1, n: int = 1,
                   skip_first: int = 0, q: float | None = None, win_s: float = 0.0,
                   fs: float = 10.0) -> pd.DataFrame:
    """윈도우 점수를 세그먼트(burst)별 경보 표로 집계한다.

    반환 컬럼: seg_uid, n_windows, n_used, alarm, first_alarm_t, delay_in_window_s. 세그먼트 순서는 입력에서
    처음 등장한 순서이고, 세그먼트 안은 t_start 오름차순으로 정렬해 처리한다.
    - skip_first=m: 세그먼트의 앞 m개 윈도우는 판정에서 제외(n_used = n_windows - m). n_used <= 0이면 행은
      남기되 alarm=False, n_used=0이다. 이런 세그먼트는 분모에서 빠지지 않고 경보 없음(정상 TN·이상 FN)으로 집계된다
      (`segment_confusion`) — 실시간 시스템에서도 그 세그먼트는 경보가 나지 않기 때문이며, 윈도우 1개짜리 이상 세그먼트
      19처럼 알려진 난제 양성을 분모에서 지우지 않기 위해서다.
    - rule="any": 사용 윈도우 중 하나라도 score > thr. skip_first=0이면 `fpr_segment`·`detection_delay_s`와 같다.
    - rule="kofn": 사용 윈도우의 초과 시퀀스에 `kofn_alarm(exceed, k, n)`을 적용해 한 번이라도 True면 경보.
      first_alarm_t는 kofn이 처음 True가 된 윈도우의 t_start.
    - rule="quantile": 사용 윈도우 점수의 q분위수 > thr이면 경보(q 필수). 세그먼트가 끝나야 판정이 나오므로
      first_alarm_t는 세그먼트 마지막 윈도우의 t_start.
    - delay_in_window_s = first_alarm_t + max(win_s - 1/fs, 0) (`detection_delay_s`와 같은 정의). 미경보는 nan.
    """
    if rule not in ("any", "kofn", "quantile"):
        raise ValueError(f"rule은 'any'·'kofn'·'quantile' 중 하나여야 한다: {rule!r}")
    if rule == "quantile" and q is None:
        raise ValueError("rule='quantile'에는 q가 필요하다")
    t = pd.DataFrame({"s": np.asarray(score, dtype=float), "u": np.asarray(seg_uid),
                      "t": np.asarray(t_start, dtype=float)})
    offset = max(win_s - 1.0 / fs, 0.0)
    rows = []
    for uid, g in t.groupby("u", sort=False):
        g = g.sort_values("t")
        n_win = len(g)
        used = g.iloc[skip_first:] if skip_first > 0 else g
        n_used = len(used)
        alarm, first_t = False, np.nan
        if n_used > 0:
            exceed = (used["s"] > thr).reset_index(drop=True)
            if rule == "any":
                if exceed.any():
                    alarm, first_t = True, float(used["t"].iloc[int(exceed.values.argmax())])
            elif rule == "kofn":
                fired = kofn_alarm(exceed, k, n)
                if fired.any():
                    alarm, first_t = True, float(used["t"].iloc[int(fired.values.argmax())])
            else:
                if float(np.quantile(used["s"].to_numpy(), q)) > thr:
                    alarm, first_t = True, float(used["t"].iloc[-1])
        rows.append({"seg_uid": uid, "n_windows": n_win, "n_used": n_used, "alarm": bool(alarm),
                     "first_alarm_t": first_t,
                     "delay_in_window_s": first_t + offset if alarm else np.nan})
    return pd.DataFrame(rows, columns=["seg_uid", "n_windows", "n_used", "alarm", "first_alarm_t",
                                       "delay_in_window_s"])


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def segment_confusion(alarms_normal: pd.DataFrame, alarms_outlier: pd.DataFrame,
                      n_outlier_total: int | None = None) -> dict:
    """`segment_alarms` 결과(정상·이상)로 세그먼트 단위 혼동행렬과 F1을 낸다.

    분모는 윈도우가 1개 이상인 세그먼트 전부(n_windows > 0 = assessable; 기존 fpr_segment·n_out_seg와 같은 분모).
    `skip_first` 등으로 n_used가 0이 된 세그먼트도 분모에 남고 경보 없음으로 센다(규칙이 판정 대상 집합을 바꾸지 않게).
    0 나눗셈은 nan. 반환 키: tp, fp, fn, tn, n_normal, n_outlier, precision, recall, f1, fpr_segment(= fp / n_normal).
    n_outlier_total은 **해당 test split에 배정된 이상 세그먼트 수(판정 불가 포함)** — gkf는 그 fold에 배정된 이상 세그먼트
    수(`split.fold_assignment` 기준, seed 42에서 fold별 5·4·4·4·4), time_block은 21. 주면 판정 불가 이상을 FN으로 센
    fn_all = fn + (n_outlier_total - n_outlier), recall_all, f1_all도 낸다(안 주면 nan). n_outlier_total < n_outlier면 ValueError.
    """
    an = alarms_normal[alarms_normal["n_windows"] > 0]
    ao = alarms_outlier[alarms_outlier["n_windows"] > 0]
    fp = int(an["alarm"].sum())
    tn = int(len(an) - fp)
    tp = int(ao["alarm"].sum())
    fn = int(len(ao) - tp)
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_normal": int(len(an)), "n_outlier": int(len(ao)),
           "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
           "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "fpr_segment": _safe_div(fp, len(an))}
    if n_outlier_total is None:
        out.update({"fn_all": float("nan"), "recall_all": float("nan"), "f1_all": float("nan")})
    else:
        if int(n_outlier_total) < len(ao):
            raise ValueError(f"n_outlier_total({n_outlier_total})이 판정 가능 이상 세그먼트 수({len(ao)})보다 작다 — split별 이상 세그먼트 수를 넣을 것")
        fn_all = fn + (int(n_outlier_total) - len(ao))
        out.update({"fn_all": int(fn_all), "recall_all": _safe_div(tp, tp + fn_all),
                    "f1_all": _safe_div(2 * tp, 2 * tp + fp + fn_all)})
    return out


def f1_vs_threshold(score_train_normal, score_test_normal, seg_test_normal, t_test_normal,
                    score_outlier, seg_outlier, t_outlier, qs=(0.99, 0.995, 0.999),
                    n_outlier_total: int | None = None, **alarm_kwargs) -> pd.DataFrame:
    """임계 수준(train 정상 분위수 q)별 세그먼트 F1 곡선.

    각 q마다 `threshold_from_normal(score_train_normal, q)`로 임계를 만들고, test 정상·이상 윈도우에
    `segment_alarms(..., **alarm_kwargs)`를 적용해 `segment_confusion` 결과를 한 행으로 쌓는다.
    컬럼: q, thr, confusion 키 전부, delay_median_s(탐지된 이상 세그먼트 delay_in_window_s의 중앙값, 없으면 nan).
    임계 수준별 F1을 보고할 뿐 **F1이 최대인 임계를 고르거나 반환하지 않는다**(이상 1이벤트 과적합 방지;
    운영점은 정상만으로 정한 허용 오경보로 정한다).
    """
    rows = []
    for q in qs:
        thr = threshold_from_normal(score_train_normal, q)
        an = segment_alarms(score_test_normal, seg_test_normal, t_test_normal, thr, **alarm_kwargs)
        ao = segment_alarms(score_outlier, seg_outlier, t_outlier, thr, **alarm_kwargs)
        row = {"q": q, "thr": thr, **segment_confusion(an, ao, n_outlier_total)}
        d = ao.loc[ao["alarm"], "delay_in_window_s"]
        row["delay_median_s"] = float(d.median()) if len(d) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def pool_confusion(confusions: list[dict]) -> dict:
    """fold별 `segment_confusion` 결과를 tp·fp·fn·tn(·fn_all) 합산 후 F1을 다시 계산한다(micro 합산).

    gkf처럼 fold마다 서로 다른 세그먼트를 한 번씩 평가하는 분할에 쓴다(이상 17·정상 530이 각각 정확히 한 번 들어감).
    time_block은 이상 전체가 블록마다 반복되므로 합산하면 TP가 블록 수만큼 부풀어 **쓰지 않는다**(블록별로 본다).
    """
    tp = sum(c["tp"] for c in confusions); fp = sum(c["fp"] for c in confusions)
    fn = sum(c["fn"] for c in confusions); tn = sum(c["tn"] for c in confusions)
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_normal": fp + tn, "n_outlier": tp + fn,
           "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
           "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "fpr_segment": _safe_div(fp, fp + tn)}
    fa = [c.get("fn_all") for c in confusions]
    if all(v == v for v in fa):   # nan 없음
        fn_all = int(sum(fa))
        out.update({"fn_all": fn_all, "recall_all": _safe_div(tp, tp + fn_all), "f1_all": _safe_div(2 * tp, 2 * tp + fp + fn_all)})
    else:
        out.update({"fn_all": float("nan"), "recall_all": float("nan"), "f1_all": float("nan")})
    return out


def threshold_for_segment_fpr(score_train_normal, seg_train_normal, t_train_normal, target_fpr: float,
                              **alarm_kwargs) -> float:
    """train **정상**만으로 운영점을 만든다: 주어진 집계 규칙(`segment_alarms` 인자)에서 train 정상의 세그먼트
    오경보율이 target_fpr 이하가 되는 가장 낮은 임계.

    q 분위수(`threshold_from_normal`)는 윈도우 단위라 규칙(kofn·quantile·skip_first)마다 세그먼트 오경보율이 달라지므로,
    규칙끼리 비교할 때는 이 함수로 세그먼트 오경보율을 같게 맞춘 뒤 recall·F1을 본다. 이상 데이터는 쓰지 않는다.
    세그먼트 오경보율은 임계에 대해 단조 비증가이므로 고유 점수값 위에서 이진 탐색한다. 어떤 임계로도 목표를 못 맞추면
    최대 점수 + 0을 돌려준다(경보 0).
    """
    s = np.asarray(score_train_normal, dtype=float)
    cand = np.unique(s[~np.isnan(s)])
    if len(cand) == 0:
        return float("nan")

    def seg_fpr(thr):
        a = segment_alarms(s, seg_train_normal, t_train_normal, thr, **alarm_kwargs)
        a = a[a["n_windows"] > 0]
        return float(a["alarm"].mean()) if len(a) else 0.0

    lo, hi = 0, len(cand) - 1
    if seg_fpr(cand[hi]) > target_fpr:
        return float(cand[hi])
    while lo < hi:                       # 최소 idx with seg_fpr(cand[idx]) <= target
        mid = (lo + hi) // 2
        if seg_fpr(cand[mid]) <= target_fpr:
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo])
