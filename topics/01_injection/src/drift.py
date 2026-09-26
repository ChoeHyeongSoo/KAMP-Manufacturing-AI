"""주제 ① 사출성형 데이터 — 드리프트·시간 누수·자기상관·cn7/rg3 비교.

02_diagnosis_deep.ipynb 의 A-2 ③⑥⑦ 항목에서 사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew, kurtosis

import data_quality as dq


# ---------------------------------------------------------------------------
# ③ 드리프트 · 시간 누수
# ---------------------------------------------------------------------------

def _psi(base: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """기준 블록(base) 대비 PSI.

    - 연속형: base 분위수로 내부 경계를 잡고 양 끝은 -inf/+inf로 열어 둔다
      (base 범위 밖 값이 버려지지 않도록 — 드리프트 신호가 바로 그 범위 밖 값이다).
    - 이산형(base 고유값 <= bins): base·cur 값의 합집합을 범주로 쓴다
      (분위수 경계가 붕괴해 PSI=0 으로 떨어지는 것을 막는다).
    - 0-count 구간은 eps 비율로 바닥 처리한다(값이 커질수록 eps에 포화되므로 PSI>~10은 순위 비교용이 아님).
    """
    base = np.asarray(base)
    cur = np.asarray(cur)
    ub = np.unique(base)
    if len(ub) <= bins:
        cats = np.union1d(ub, np.unique(cur))
        b_cnt = np.array([(base == v).sum() for v in cats])
        c_cnt = np.array([(cur == v).sum() for v in cats])
    else:
        inner = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))[1:-1]
        edges = np.concatenate(([-np.inf], inner, [np.inf]))
        b_cnt, _ = np.histogram(base, bins=edges)
        c_cnt, _ = np.histogram(cur, bins=edges)
    if len(b_cnt) < 2:
        return 0.0
    b_pct = np.clip(b_cnt / b_cnt.sum(), eps, None)
    c_pct = np.clip(c_cnt / c_cnt.sum(), eps, None)
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def block_stats(X: pd.DataFrame, n_blocks: int = 5) -> pd.DataFrame:
    """행 순서를 n_blocks개로 나눠 블록별 평균 · KS(블록0 대비) · PSI(블록0 대비)를 계산한다."""
    n = len(X)
    bounds = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = [X.iloc[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]

    rows = []
    for c in X.columns:
        base = blocks[0][c].to_numpy()
        rec = {"var": c}
        for i, b in enumerate(blocks):
            rec[f"mean_b{i}"] = float(b[c].mean())
        max_psi, min_p = 0.0, 1.0
        for i in range(1, n_blocks):
            cur = blocks[i][c].to_numpy()
            stat, p = ks_2samp(base, cur)
            psi = _psi(base, cur)
            rec[f"ks_p_b{i}"] = p
            rec[f"psi_b{i}"] = psi
            max_psi = max(max_psi, psi)
            min_p = min(min_p, p)
        rec["max_psi"] = max_psi
        rec["min_ks_p"] = min_p
        rows.append(rec)
    return pd.DataFrame(rows).set_index("var").sort_values("max_psi", ascending=False)


def auc_by_split(X: pd.DataFrame, y: pd.Series, var: str, time_test_idx, n_splits: int = 5,
                  random_state: int = 0) -> dict:
    """단일 변수의 랜덤 5-fold AUC(평균) vs 시간 블록(test=time_test_idx) AUC 비교."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rand_aucs = []
    for _, te in skf.split(X, y):
        yt = y.iloc[te]
        if yt.nunique() < 2:
            continue
        a = roc_auc_score(yt, X[var].iloc[te])
        rand_aucs.append(max(a, 1 - a))

    time_mask = X.index.isin(time_test_idx)
    yt = y[time_mask]
    auc_time = np.nan
    if yt.nunique() == 2:
        a = roc_auc_score(yt, X.loc[time_mask, var])
        auc_time = max(a, 1 - a)

    return {"var": var, "auc_random_mean": float(np.mean(rand_aucs)) if rand_aucs else np.nan,
            "auc_random_std": float(np.std(rand_aucs)) if rand_aucs else np.nan,
            "auc_time_block": auc_time, "n_time_test": int(time_mask.sum()), "n_time_test_pos": int(yt.sum())}


def simultaneous_extreme_rows(X: pd.DataFrame, cols: list[str], z: float = 3.0, k: int | None = None) -> pd.DataFrame:
    """cols 중 k개 이상이 동시에 |z|>z 인 행을 찾는다 (기본 k=len(cols))."""
    if k is None:
        k = len(cols)
    ext = X[cols].abs() > z
    cnt = ext.sum(axis=1)
    sel = cnt[cnt >= k]
    return pd.DataFrame({"idx": sel.index, "n_extreme": sel.to_numpy()})


# ---------------------------------------------------------------------------
# ⑥ 샷 간 자기상관 · 불량 run-length
# ---------------------------------------------------------------------------

def autocorr_table(X: pd.DataFrame, lags: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    """행 순서(=수집 순서로 간주) 기준 lag별 자기상관. 변수별로 계산한다."""
    rows = []
    for c in X.columns:
        s = X[c]
        rec = {"var": c}
        for lag in lags:
            rec[f"lag{lag}"] = s.autocorr(lag=lag)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("var").sort_values(f"lag{lags[0]}", ascending=False)


def fail_run_lengths(y: pd.Series) -> pd.Series:
    """불량(y==1) 연속 run 길이 분포. data_quality.run_lengths 재사용."""
    return dq.run_lengths(y == 1)


# ---------------------------------------------------------------------------
# ⑦ cn7 · rg3 결합 가능성
# ---------------------------------------------------------------------------

def shape_compare(Xa: pd.DataFrame, Xb: pd.DataFrame, common_cols: list[str] | None = None) -> pd.DataFrame:
    """두 파일(z-score라 형태만 비교)의 변수별 KS 통계량·왜도·첨도를 비교한다."""
    cols = common_cols or [c for c in Xa.columns if c in Xb.columns]
    rows = []
    for c in cols:
        a, b = Xa[c].to_numpy(), Xb[c].to_numpy()
        stat, p = ks_2samp(a, b)
        rows.append({"var": c, "ks_stat": stat, "ks_p": p,
                      "skew_cn7": skew(a), "skew_rg3": skew(b),
                      "kurt_cn7": kurtosis(a), "kurt_rg3": kurtosis(b)})
    return pd.DataFrame(rows).set_index("var").sort_values("ks_stat", ascending=False)
