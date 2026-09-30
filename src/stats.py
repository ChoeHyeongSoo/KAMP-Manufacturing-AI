"""주제 ③ 프레스 유압펌프 — 모델링 전 통계 검정·요약 도구 (그룹 비교 · 상관/VIF · PCA · 분포 · 추이 · 부트스트랩).

12 노트북(피처 통계)과 04(추이·드리프트)가 같은 정의로 표를 만들도록 고정한다. 모든 함수는
DataFrame을 받아 DataFrame(표)을 돌려주므로 노트북에서 바로 `to_csv` 할 수 있다.

규칙
- **윈도우 단위 p값 보고 금지, 세그먼트 대표값으로 검정.** 한 세그먼트(burst)의 윈도우는 서로 거의 같은
  값이라 독립 표본이 아니다. 검정·추이·상관의 p값은 `segment_representatives`로 세그먼트 1행으로 줄인
  표에서만 계산한다. 검정 함수는 호출자가 넘긴 표를 그대로 한 행 = 한 표본으로 취급한다.
- 임계·표준화·PCA는 train 정상 표본으로만 적합한다(평가 표본으로 적합하지 않는다).
- 성능 지표의 신뢰구간은 윈도우가 아니라 **세그먼트를 재표집 단위**로 하는 부트스트랩으로 구한다.
- 난수는 `seed`(기본 42)로 `np.random.default_rng`에서만 만든다.
- 의존성은 numpy·pandas·scipy·scikit-learn 뿐이다(VIF·Mann–Kendall은 직접 구현).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluate import auc as _auc

SEED = 42
KEEP_COLS = ("src", "label", "fold", "block", "state", "vib_grade", "cur_grade")


# --- 1) 세그먼트 대표값 --------------------------------------------------

def segment_representatives(df: pd.DataFrame, cols, key: str = "seg_uid", keep=KEEP_COLS,
                            agg: str = "median") -> pd.DataFrame:
    """윈도우 표를 세그먼트 1행으로 축약한다. `cols`는 agg(기본 중앙값), `keep` 열은 first.

    `keep` 중 표에 없는 열은 건너뛴다. `n_windows`(세그먼트의 윈도우 수) 열을 추가한다.
    """
    cols = list(cols)
    keep = [c for c in keep if c in df.columns and c != key and c not in cols]
    g = df.groupby(key, sort=False)
    out = g[cols].agg(agg)
    if keep:
        out = out.join(g[keep].first())
    out["n_windows"] = g.size()
    return out.reset_index()


# --- 2) 그룹 비교 --------------------------------------------------------

def bh_adjust(p) -> np.ndarray:
    """Benjamini–Hochberg 보정 p값. nan은 nan으로 두고 나머지 개수만 m으로 센다."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _two_sample(a: np.ndarray, b: np.ndarray) -> dict:
    """한 열의 a vs b 통계량 dict (n<3이면 검정값 nan)."""
    r = {"n_a": len(a), "n_b": len(b),
         "mean_a": a.mean() if len(a) else np.nan, "mean_b": b.mean() if len(b) else np.nan,
         "median_a": np.median(a) if len(a) else np.nan, "median_b": np.median(b) if len(b) else np.nan,
         "welch_t": np.nan, "welch_p": np.nan, "mwu_u": np.nan, "mwu_p": np.nan,
         "cohen_d": np.nan, "cliff_delta": np.nan, "median_b_over_q99_a": np.nan}
    if len(a) >= 1 and len(b):
        q99 = np.quantile(a, 0.99)
        if q99 != 0:
            r["median_b_over_q99_a"] = float(np.median(b) / q99)
    if len(a) < 3 or len(b) < 3:
        return r
    try:
        t = st.ttest_ind(b, a, equal_var=False)
        r["welch_t"], r["welch_p"] = float(t.statistic), float(t.pvalue)
    except Exception:
        pass
    try:
        u = st.mannwhitneyu(a, b, alternative="two-sided")
        r["mwu_u"], r["mwu_p"] = float(u.statistic), float(u.pvalue)
        r["cliff_delta"] = float(1.0 - 2.0 * u.statistic / (len(a) * len(b)))   # b가 클수록 +
    except Exception:
        pass
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp > 0:
        r["cohen_d"] = float((b.mean() - a.mean()) / sp)
    return r


def group_test_table(df: pd.DataFrame, cols, group_col: str, groups=(0, 1), alpha: float = 0.05) -> pd.DataFrame:
    """열별 groups[0](a) vs groups[1](b) 비교표. **df는 세그먼트 대표값 표여야 한다.**

    컬럼: feature, n_a, n_b, mean_a/b, median_a/b, welch_t/p, mwu_u/p, cohen_d(pooled sd, b−a),
    cliff_delta(P(b>a)−P(b<a)), p_adj_bh(mwu_p의 BH 보정), significant(p_adj_bh<alpha),
    median_b_over_q99_a(b 중앙값 / a의 q99; q99가 0이면 nan). nan은 열별로 제거 후 계산한다.
    """
    ga, gb = groups
    rows = []
    for c in cols:
        a = df.loc[df[group_col] == ga, c].dropna().to_numpy(dtype=float)
        b = df.loc[df[group_col] == gb, c].dropna().to_numpy(dtype=float)
        rows.append({"feature": c, **_two_sample(a, b)})
    out = pd.DataFrame(rows)
    out["p_adj_bh"] = bh_adjust(out["mwu_p"].to_numpy())
    out["significant"] = out["p_adj_bh"] < alpha
    order = ["feature", "n_a", "n_b", "mean_a", "mean_b", "median_a", "median_b", "welch_t", "welch_p",
             "mwu_u", "mwu_p", "cohen_d", "cliff_delta", "p_adj_bh", "significant", "median_b_over_q99_a"]
    return out[order]


# --- 3) 상관 군집 · VIF ---------------------------------------------------

def corr_clusters(df: pd.DataFrame, cols, method: str = "spearman", thr: float = 0.9
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """상관행렬과 |ρ|≥thr 계층 군집(거리 1−|ρ|, complete linkage, 거리 임계 1−thr).

    반환: (corr 정방행렬, clusters). clusters 컬럼: feature, cluster, cluster_size, representative
    (군집 내 다른 열과의 평균 |ρ|가 최대인 열; 크기 1이면 자기 자신), is_representative.
    상수 열처럼 상관이 nan이면 0으로 두고 대각은 1로 둔다.
    """
    cols = list(cols)
    corr = df[cols].corr(method=method)
    a = np.abs(corr.to_numpy(dtype=float))
    a = np.nan_to_num(a, nan=0.0)
    np.fill_diagonal(a, 1.0)
    if len(cols) == 1:
        labels = np.array([1])
    else:
        dist = np.clip(1.0 - a, 0.0, None)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform(dist, checks=False), method="complete")
        labels = fcluster(z, t=1.0 - thr + 1e-12, criterion="distance")
    rows = []
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        if len(idx) == 1:
            rep = idx[0]
        else:
            sub = a[np.ix_(idx, idx)]
            rep = idx[int(np.argmax((sub.sum(axis=1) - 1.0) / (len(idx) - 1)))]
        for i in idx:
            rows.append({"feature": cols[i], "cluster": int(k), "cluster_size": len(idx),
                         "representative": cols[rep], "is_representative": bool(i == rep)})
    clusters = pd.DataFrame(rows).set_index("feature").loc[cols].reset_index()
    return corr, clusters


def vif_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    """VIF와 표준화 행렬의 조건수. 각 열을 나머지 열에 OLS(lstsq) 회귀해 1/(1−R²).

    R²≥1−1e-12면 inf. 분산 0인 열의 VIF는 nan이며 조건수 계산에서 제외한다.
    컬럼: feature, vif, condition_number(모든 행 동일 값). nan 행은 제거한다.
    """
    cols = list(cols)
    x = df[cols].dropna().to_numpy(dtype=float)
    sd = x.std(axis=0, ddof=0)
    live = sd > 0
    z = (x - x.mean(axis=0)) / np.where(live, sd, 1.0)
    vifs = np.full(len(cols), np.nan)
    live_idx = np.flatnonzero(live)
    for j in live_idx:
        others = [i for i in live_idx if i != j]
        if not others:
            vifs[j] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(z[:, others], z[:, j], rcond=None)
        resid = z[:, j] - z[:, others] @ beta
        r2 = 1.0 - float(resid @ resid) / float(z[:, j] @ z[:, j])
        vifs[j] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    if len(live_idx) >= 1:
        s = np.linalg.svd(z[:, live_idx], compute_uv=False)
        cond = float(s.max() / s.min()) if s.min() > 0 else float("inf")
    else:
        cond = float("nan")
    return pd.DataFrame({"feature": cols, "vif": vifs, "condition_number": cond})


# --- 4) PCA ---------------------------------------------------------------

def pca_fit_report(X_train: pd.DataFrame, cols, var_targets=(0.9, 0.95), n_components=None) -> dict:
    """train 정상 표본으로 StandardScaler+PCA를 적합하고 요약을 dict로 돌려준다.

    반환 키: scaler, pca, explained(component, explained_ratio, cumulative),
    n_for({목표 분산: 필요한 성분 수}), loadings(index=cols, columns=PC1..PCk).
    nan 행은 적합 전에 제거한다.
    """
    cols = list(cols)
    x = X_train[cols].dropna().to_numpy(dtype=float)
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=n_components).fit(scaler.transform(x))
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    k = len(ratio)
    explained = pd.DataFrame({"component": np.arange(1, k + 1), "explained_ratio": ratio, "cumulative": cum})
    n_for = {t: int(min(np.searchsorted(cum, t - 1e-12) + 1, k)) for t in var_targets}
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=[f"PC{i}" for i in range(1, k + 1)])
    return {"scaler": scaler, "pca": pca, "explained": explained, "n_for": n_for, "loadings": loadings}


def pca_recon_error(report: dict, X) -> np.ndarray:
    """표준화 → 투영 → 역변환한 뒤 행별 RMSE(표준화 공간). nan이 있는 행은 nan."""
    scaler, pca = report["scaler"], report["pca"]
    x = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X, dtype=float)
    ok = ~np.isnan(x).any(axis=1)
    out = np.full(len(x), np.nan)
    if ok.any():
        z = scaler.transform(x[ok])
        rec = pca.inverse_transform(pca.transform(z))
        out[ok] = np.sqrt(np.mean((z - rec) ** 2, axis=1))
    return out


# --- 5) 분포 · 분위수 안정성 -----------------------------------------------

def distribution_summary(df: pd.DataFrame, cols, transforms=("none", "log1p"), seed: int = SEED) -> pd.DataFrame:
    """열×변환별 분포 요약. 컬럼: feature, transform, n, mean, std, skew, kurt(초과), shapiro_p, shapiro_n, q99.

    log1p는 최소값이 −1 이하, log는 최소값이 0 이하이면 값 열이 nan인 행. Shapiro는 표본 5,000 초과 시
    `default_rng(seed)`로 5,000개를 뽑아 계산하고 사용 표본 수를 `shapiro_n`에 적는다.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in cols:
        x = df[c].dropna().to_numpy(dtype=float)
        for tr in transforms:
            if tr not in ("none", "log1p", "log"):
                raise ValueError(f"알 수 없는 변환: {tr}")
            row = {"feature": c, "transform": tr, "n": len(x), "mean": np.nan, "std": np.nan, "skew": np.nan,
                   "kurt": np.nan, "shapiro_p": np.nan, "shapiro_n": 0, "q99": np.nan}
            valid = len(x) > 0 and (tr == "none" or (tr == "log1p" and x.min() > -1) or (tr == "log" and x.min() > 0))
            if valid:
                y = x if tr == "none" else (np.log1p(x) if tr == "log1p" else np.log(x))
                row.update(mean=float(y.mean()), std=float(y.std(ddof=1)) if len(y) > 1 else np.nan,
                           q99=float(np.quantile(y, 0.99)))
                if len(y) > 3 and y.std() > 0:
                    row["skew"] = float(st.skew(y, bias=False))
                    row["kurt"] = float(st.kurtosis(y, fisher=True, bias=False))
                    ys = y if len(y) <= 5000 else rng.choice(y, size=5000, replace=False)
                    row["shapiro_p"] = float(st.shapiro(ys).pvalue)
                    row["shapiro_n"] = len(ys)
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_stability(df: pd.DataFrame, col: str, fold_col: str, q: float = 0.99,
                       label_col: str = "label", normal_value=0) -> pd.DataFrame:
    """fold별 train(해당 fold 제외) 정상 q분위수와 변동 계수. 컬럼 fold, thr; 마지막 행 fold="cv", thr=std/mean."""
    rows = []
    for f in sorted(df[fold_col].dropna().unique()):
        tr = df[(df[fold_col] != f) & (df[label_col] == normal_value)][col].dropna().to_numpy(dtype=float)
        rows.append({"fold": f, "thr": float(np.quantile(tr, q)) if len(tr) else np.nan})
    out = pd.DataFrame(rows, columns=["fold", "thr"])
    thr = out["thr"].to_numpy(dtype=float)
    m = np.nanmean(thr) if len(thr) else np.nan
    cv = float(np.nanstd(thr, ddof=1) / m) if len(thr) > 1 and m else np.nan
    out["fold"] = out["fold"].astype(object)
    out.loc[len(out)] = ["cv", cv]
    return out


# --- 6) 시간 추세 · 드리프트 -------------------------------------------------

def _to_seconds(t: pd.Series) -> np.ndarray:
    """숫자는 그대로, datetime은 초 단위 float으로 변환."""
    if pd.api.types.is_datetime64_any_dtype(t):
        return (pd.to_datetime(t).astype("int64").to_numpy() / 1e9).astype(float)
    return t.to_numpy(dtype=float)


def _mann_kendall(y: np.ndarray) -> tuple[float, float, float]:
    """시간순 y의 Mann–Kendall (S, Z, 양측 p). 동률 보정 분산과 연속성 보정 포함(정규근사)."""
    n = len(y)
    s = 0.0
    for i in range(n - 1):
        s += float(np.sign(y[i + 1:] - y[i]).sum())
    _, cnt = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - (cnt * (cnt - 1) * (2 * cnt + 5)).sum()) / 18.0
    if var <= 0:
        return s, np.nan, np.nan
    z = (s - np.sign(s)) / np.sqrt(var)
    return s, float(z), float(2 * (1 - st.norm.cdf(abs(z))))


def trend_table(df: pd.DataFrame, cols, time_col: str, method=("spearman", "mann_kendall")) -> pd.DataFrame:
    """열별 시간 추세. 컬럼: feature, n, spearman_rho/p, mk_s/z/p, slope_per_unit(Theil–Sen, 시간 1단위당).

    `method`에 없는 검정 열은 nan. time_col은 숫자 또는 datetime(초로 변환). 세그먼트 대표값 표를 넣는다.
    """
    t_all = _to_seconds(df[time_col])
    rows = []
    for c in cols:
        y_all = df[c].to_numpy(dtype=float)
        ok = ~(np.isnan(y_all) | np.isnan(t_all))
        t, y = t_all[ok], y_all[ok]
        order = np.argsort(t, kind="stable")
        t, y = t[order], y[order]
        row = {"feature": c, "n": len(y), "spearman_rho": np.nan, "spearman_p": np.nan,
               "mk_s": np.nan, "mk_z": np.nan, "mk_p": np.nan, "slope_per_unit": np.nan}
        if len(y) >= 3 and np.ptp(y) > 0 and np.ptp(t) > 0:
            if "spearman" in method:
                r = st.spearmanr(t, y)
                row["spearman_rho"], row["spearman_p"] = float(r.statistic), float(r.pvalue)
            if "mann_kendall" in method:
                row["mk_s"], row["mk_z"], row["mk_p"] = _mann_kendall(y)
            row["slope_per_unit"] = float(st.theilslopes(y, t)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def drift_ks_table(df: pd.DataFrame, cols, block_col: str, blocks=None) -> pd.DataFrame:
    """블록 k(k≥1)의 표본 vs 블록 <k 표본의 KS 통계·p와 Levene p.

    컬럼: feature, block, n_ref, n_blk, ks_stat, ks_p, levene_p, median_ref, median_blk.
    blocks는 시간순 블록 값 목록(기본: 정렬된 고유값). 첫 블록은 기준만 되고 행이 없다.
    """
    blocks = list(sorted(df[block_col].dropna().unique())) if blocks is None else list(blocks)
    rows = []
    for c in cols:
        for i in range(1, len(blocks)):
            ref = df.loc[df[block_col].isin(blocks[:i]), c].dropna().to_numpy(dtype=float)
            cur = df.loc[df[block_col] == blocks[i], c].dropna().to_numpy(dtype=float)
            row = {"feature": c, "block": blocks[i], "n_ref": len(ref), "n_blk": len(cur),
                   "ks_stat": np.nan, "ks_p": np.nan, "levene_p": np.nan,
                   "median_ref": np.median(ref) if len(ref) else np.nan,
                   "median_blk": np.median(cur) if len(cur) else np.nan}
            if len(ref) >= 2 and len(cur) >= 2:
                k = st.ks_2samp(ref, cur)
                row["ks_stat"], row["ks_p"] = float(k.statistic), float(k.pvalue)
                try:
                    row["levene_p"] = float(st.levene(ref, cur).pvalue)
                except Exception:
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


# --- 7) 세그먼트 단위 부트스트랩 ----------------------------------------------

def _seg_index(y, groups):
    """세그먼트별 행 위치 목록과 세그먼트 라벨(0/1)."""
    y = np.asarray(y).astype(int)
    g = pd.Series(np.asarray(groups))
    idx = g.groupby(g.to_numpy(), sort=False).indices
    seg_rows = list(idx.values())
    seg_lab = np.array([int(round(y[r].mean())) for r in seg_rows])
    return seg_rows, seg_lab


def _ci(x, ci):
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return float(np.quantile(x, lo)), float(np.quantile(x, hi))


def _resample(rng, seg_rows, pos_ids, neg_ids):
    """정상·이상 세그먼트를 각각 복원 추출하고 (뽑힌 세그먼트 번호, 모든 윈도우 행 위치)를 돌려준다."""
    pick = np.concatenate([rng.choice(neg_ids, size=len(neg_ids), replace=True),
                           rng.choice(pos_ids, size=len(pos_ids), replace=True)])
    return pick, np.concatenate([seg_rows[i] for i in pick])


def bootstrap_auc_ci(y, score, groups, n_boot: int = 1000, seed: int = SEED, ci: float = 0.95,
                     thr: float | None = None) -> dict:
    """세그먼트 단위 복원 추출 부트스트랩 AUC(·fpr_segment) 신뢰구간.

    정상·이상 세그먼트 집합을 각각 복원 추출하고 뽑힌 세그먼트의 모든 윈도우로 AUC를 계산한다.
    thr가 있으면 fpr_segment(정상 세그먼트 중 초과 윈도우가 하나라도 있는 비율)도 계산한다 —
    같은 세그먼트가 중복 추출되면 중복만큼 가중한다(evaluate.fpr_segment와 점추정은 동일).
    클래스가 하나뿐인 재표집은 건너뛰고 `n_valid`에 유효 재표집 수를 적는다.
    반환: auc, auc_ci_low/high, [fpr_segment, fpr_segment_ci_low/high], n_boot, n_valid, n_seg_normal, n_seg_outlier.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    seg_rows, seg_lab = _seg_index(y, groups)
    pos_ids, neg_ids = np.flatnonzero(seg_lab == 1), np.flatnonzero(seg_lab == 0)
    out = {"auc": _auc(y, score), "n_boot": n_boot, "n_seg_normal": len(neg_ids), "n_seg_outlier": len(pos_ids)}
    exceed = None
    if thr is not None:
        exceed = np.array([bool((score[r] > thr).any()) for r in seg_rows])
        out["fpr_segment"] = float(exceed[neg_ids].mean()) if len(neg_ids) else float("nan")
    aucs, fprs = [], []
    if len(pos_ids) and len(neg_ids):
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            pick, rows = _resample(rng, seg_rows, pos_ids, neg_ids)
            a = _auc(y[rows], score[rows])
            if np.isnan(a):
                continue
            aucs.append(a)
            if exceed is not None:
                fprs.append(exceed[pick[:len(neg_ids)]].mean())
    out["n_valid"] = len(aucs)
    out["auc_ci_low"], out["auc_ci_high"] = _ci(aucs, ci) if aucs else (float("nan"),) * 2
    if thr is not None:
        out["fpr_segment_ci_low"], out["fpr_segment_ci_high"] = _ci(fprs, ci) if fprs else (float("nan"),) * 2
    return out


def bootstrap_auc_diff(y, score_a, score_b, groups, n_boot: int = 1000, seed: int = SEED, ci: float = 0.95) -> dict:
    """같은 세그먼트 재표집 안에서 AUC(b) − AUC(a)의 점추정·구간.

    반환: diff, ci_low, ci_high, p_two_sided(부트스트랩 분포에서 0을 기준으로 한 양측 근사 =
    2·min(P(d≤0), P(d≥0)), 최소 1/n_valid), overlap_zero(구간이 0을 포함하면 True), n_valid.
    """
    y = np.asarray(y).astype(int)
    sa, sb = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    seg_rows, seg_lab = _seg_index(y, groups)
    pos_ids, neg_ids = np.flatnonzero(seg_lab == 1), np.flatnonzero(seg_lab == 0)
    diff = _auc(y, sb) - _auc(y, sa)
    ds = []
    if len(pos_ids) and len(neg_ids):
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            _, rows = _resample(rng, seg_rows, pos_ids, neg_ids)
            a, b = _auc(y[rows], sa[rows]), _auc(y[rows], sb[rows])
            if not (np.isnan(a) or np.isnan(b)):
                ds.append(b - a)
    if not ds:
        return {"diff": diff, "ci_low": np.nan, "ci_high": np.nan, "p_two_sided": np.nan,
                "overlap_zero": True, "n_valid": 0}
    ds = np.asarray(ds)
    lo, hi = _ci(ds, ci)
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return {"diff": diff, "ci_low": lo, "ci_high": hi, "p_two_sided": float(max(p, 1.0 / len(ds))),
            "overlap_zero": bool(lo <= 0 <= hi), "n_valid": len(ds)}
