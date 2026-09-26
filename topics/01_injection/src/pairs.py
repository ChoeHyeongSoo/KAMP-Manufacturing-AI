"""주제 ① 사출성형 데이터 — 쌍(pair) 구조, 타깃 정의 비교, labeled<->unlabeled 매칭.

02_diagnosis_deep.ipynb 의 A-2 ①④⑤ 항목에서 사용한다.
data_quality.py(01 노트북에서 사용)의 로딩·row_key 로직을 그대로 재사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest

import data_quality as dq

LABEL = dq.LABEL


# ---------------------------------------------------------------------------
# ① 쌍 구조 규명
# ---------------------------------------------------------------------------

def pair_table(df: pd.DataFrame, label_col: str = LABEL) -> pd.DataFrame:
    """X-키(24변수, 8자리 반올림)로 그룹화해 쌍(및 예외) 단위 테이블을 만든다.

    반환 컬럼: key, group_size, idx1, idx2, gap(=idx2-idx1), adjacent(gap==1),
    exact_equal(반올림 없이 24변수 완전 일치 여부), label1, label2, pattern(00/01/11/single)
    """
    has_label = label_col in df.columns
    X = df.drop(columns=[label_col]) if has_label else df
    y = df[label_col] if has_label else None
    key = dq.row_key(X)

    rows = []
    for k, sub_idx in df.index.to_series().groupby(key.values):
        idxs = sorted(sub_idx.tolist())
        n = len(idxs)
        rec = {"key": k, "group_size": n}
        if n == 1:
            i1 = idxs[0]
            rec.update(idx1=i1, idx2=np.nan, gap=np.nan, adjacent=np.nan, exact_equal=np.nan,
                       label1=(int(y.loc[i1]) if has_label else np.nan), label2=np.nan,
                       pattern="single")
        else:
            i1, i2 = idxs[0], idxs[-1]
            gap = i2 - i1
            exact_equal = bool((X.loc[i1].to_numpy() == X.loc[i2].to_numpy()).all())
            if has_label:
                l1, l2 = int(y.loc[i1]), int(y.loc[i2])
                pattern = "".join(sorted([str(l1), str(l2)]))
            else:
                l1 = l2 = np.nan
                pattern = np.nan
            rec.update(idx1=i1, idx2=i2, gap=gap, adjacent=bool(gap == 1), exact_equal=exact_equal,
                       label1=l1, label2=l2, pattern=pattern)
        rows.append(rec)
    return pd.DataFrame(rows)


def duplicate_gap_stats(df: pd.DataFrame, label_col: str | None = LABEL) -> pd.DataFrame:
    """같은 X(24변수, 8자리 반올림 키)를 가진 행들의 인덱스 간 연속 간격(gap) 분포.

    labeled의 "쌍"(그룹 크기 2)뿐 아니라 unlabeled처럼 그룹 크기가 2보다 클 수 있는 완전 중복도 다루기 위해,
    정렬된 인덱스의 연속 차분(diff)을 전부 모은다(그룹 크기 n이면 차분 n-1개). A-1(unlabeled 완전 중복 18%와
    labeled 쌍이 같은 메커니즘인지)의 비교용.
    """
    X = df.drop(columns=[label_col]) if label_col and label_col in df.columns else df
    key = dq.row_key(X)
    rows = []
    for k, sub_idx in df.index.to_series().groupby(key.values):
        idxs = sorted(sub_idx.tolist())
        if len(idxs) < 2:
            continue
        for g in np.diff(idxs):
            rows.append({"key": k, "group_size": len(idxs), "gap": int(g)})
    return pd.DataFrame(rows)


def order_bias_test(pt: pd.DataFrame) -> dict:
    """충돌 쌍(01)에서 불량 행이 첫 번째/두 번째 중 어디에 오는지 편향을 검정한다."""
    conflict = pt[pt["pattern"] == "01"]
    n = len(conflict)
    fail_first = int((conflict["label1"] == 1).sum())
    fail_second = int((conflict["label2"] == 1).sum())
    if n:
        p = binomtest(fail_second, n, p=0.5).pvalue
    else:
        p = np.nan
    return {"n_conflict": n, "fail_first": fail_first, "fail_second": fail_second, "binom_p": p}


def mannwhitney_conflict_vs_agree(df: pd.DataFrame, pt: pd.DataFrame, label_col: str = LABEL) -> pd.DataFrame:
    """충돌 쌍 vs 일치 쌍(00/11)의 변수별 분포 차이: Mann-Whitney U, p, 효과크기(rank-biserial)."""
    X = df.drop(columns=[label_col]) if label_col in df.columns else df
    pairs2 = pt[pt["group_size"] == 2].copy()
    rep = pd.DataFrame(index=pairs2.index)
    for c in X.columns:
        v1 = X.loc[pairs2["idx1"], c].to_numpy()
        v2 = X.loc[pairs2["idx2"], c].to_numpy()
        rep[c] = (v1 + v2) / 2.0
    is_conflict = (pairs2["pattern"] == "01").to_numpy()

    out = []
    for c in X.columns:
        a = rep.loc[is_conflict, c].to_numpy()
        b = rep.loc[~is_conflict, c].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        effect_r = 1 - (2 * stat) / (n1 * n2)  # rank-biserial correlation
        out.append({"var": c, "n_conflict": n1, "n_agree": n2, "U": stat, "p": p, "effect_r": effect_r})
    res = pd.DataFrame(out).set_index("var").sort_values("p")
    res["p_bonf"] = (res["p"] * len(res)).clip(upper=1.0)
    return res


# ---------------------------------------------------------------------------
# 타깃 정의 3안 + GroupKFold 로지스틱 F1 비교 (진단용 소형 실험)
# ---------------------------------------------------------------------------

def target_variants(df: pd.DataFrame, label_col: str = LABEL):
    """타깃 정의 후보(max / mean>=0.5 / min / 충돌 제외)에 대한 (X, y, group) 튜플을 만든다.

    그룹 크기가 항상 1~2이므로 실제로 서로 다른 이진 결정을 내리는 것은 **max와 min 둘뿐**이다
    (mean>=0.5는 max와 항상 같은 결정을 내린다 — 아래 참고).
    - max: 충돌 쌍(그룹 내 라벨이 갈리는 경우)을 **양성**으로 본다("불량 의심 포함", 재현율 우선).
    - min: 충돌 쌍을 **음성**으로 본다("두 기록이 모두 불량이어야 확정 불량", 정밀도 우선/보수적 라벨링).
    - exclude_conflict: 충돌 쌍 행을 아예 제거한다(양성이 0개로 줄어들 수 있어 불안정 — 참고용으로만 남긴다).
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    key = dq.row_key(X)
    agg = pd.DataFrame({"y": y.to_numpy(), "key": key.to_numpy()}).groupby("key")["y"].agg(
        ["max", "mean", "min", "nunique"])
    y_max = key.map(agg["max"]).astype(int)
    y_mean_bin = (key.map(agg["mean"]) >= 0.5).astype(int)
    y_min = key.map(agg["min"]).astype(int)
    is_conflict = key.map(agg["nunique"] > 1).astype(bool)

    variants = {
        "max": (X, y_max, key),
        "mean_ge_0.5": (X, y_mean_bin, key),
        "min": (X, y_min, key),
        "exclude_conflict": (X[~is_conflict], y[~is_conflict], key[~is_conflict]),
    }
    group_pos = int((agg["max"] > 0).sum())  # 그룹(샷) 단위 고유 불량 수(=max 기준)
    group_soft_pos = float(agg["mean"].sum())  # 그룹 단위 기대 양성 수(충돌=0.5로 카운트)
    group_pure_pos = int(((agg["nunique"] == 1) & (agg["max"] == 1)).sum())  # min 기준 양성 그룹 수(모호하지 않은 불량)
    meta = {"group_pos": group_pos, "group_soft_pos": group_soft_pos, "group_pure_pos": group_pure_pos,
            "n_groups": int(len(agg)), "n_conflict_groups": int((agg["nunique"] > 1).sum())}
    return variants, meta


def cv_f1_by_target(df: pd.DataFrame, label_col: str = LABEL, n_splits: int = 5, random_state: int = 0) -> pd.DataFrame:
    """타깃 정의 후보(max/mean_ge_0.5/min/exclude_conflict) 각각에 대해 GroupKFold(X-키) 5-fold
    로지스틱(class_weight='balanced') F1을 계산한다.

    모델 비교가 목적이 아니라 타깃 정의 선택을 위한 소형 진단 실험이다.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score

    variants, meta = target_variants(df, label_col)
    records = []
    for name, (X, y, groups) in variants.items():
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            records.append({"target": name, "fold": np.nan, "f1": np.nan, "n_pos": n_pos,
                             "note": "양성 또는 음성 표본 0개 - 실행 불가"})
            continue
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X.iloc[tr], y.iloc[tr])
            pred = clf.predict(X.iloc[te])
            f1 = f1_score(y.iloc[te], pred, zero_division=0)
            records.append({"target": name, "fold": fold, "f1": f1, "n_pos": n_pos, "note": ""})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ④ labeled <-> unlabeled 근사 매칭 (순위 기반)
# ---------------------------------------------------------------------------

def rank_match_to_reference(L: pd.DataFrame, U: pd.DataFrame, label_col: str = LABEL, seed: int = 0):
    """L의 각 행(순위 벡터)이 U 안에 얼마나 가까운 이웃을 갖는지 근사 매칭한다.

    파일 내부 순위(percentile rank)로 정규화한 뒤 24변수 유클리드 거리로 최근접 이웃을 찾는다.
    난수 기준선(rand_dist)과 비교해 "실제 매칭"과 "우연"을 구분한다.
    """
    from scipy.spatial import cKDTree

    Lx = L.drop(columns=[label_col]) if label_col in L.columns else L
    Ux = U.drop(columns=[label_col]) if label_col in U.columns else U
    cols = [c for c in Lx.columns if c in Ux.columns]

    L_rank = Lx[cols].rank(pct=True).to_numpy()
    U_rank = Ux[cols].rank(pct=True).to_numpy()

    tree = cKDTree(U_rank)
    dist, nn_pos = tree.query(L_rank, k=1)

    rng = np.random.default_rng(seed)
    rand_pts = rng.random(size=(len(Lx), len(cols)))
    rand_dist, _ = tree.query(rand_pts, k=1)

    return pd.DataFrame({"dist": dist, "nn_pos": nn_pos}, index=Lx.index), rand_dist


# ---------------------------------------------------------------------------
# ⑤ 변수 성격 전수 점검
# ---------------------------------------------------------------------------

def var_profile(X: pd.DataFrame, discrete_th: int = 20, spike_th: float = 0.05) -> pd.DataFrame:
    """24변수 고유값 수, 이산/연속, 상수, 스파이크 값(한 값이 spike_th 이상 차지)을 점검한다."""
    rows = []
    for c in X.columns:
        vc = X[c].value_counts(normalize=True)
        nun = int(X[c].nunique())
        top_val = float(vc.index[0])
        top_share = float(vc.iloc[0])
        rows.append({
            "var": c, "nunique": nun, "is_constant": nun == 1, "is_discrete": nun <= discrete_th,
            "top_value": top_val, "top_share": top_share, "is_spike": bool(top_share >= spike_th and nun > 1),
        })
    return pd.DataFrame(rows).set_index("var")


def value_segments(x: pd.Series) -> pd.DataFrame:
    """이산 변수 값이 바뀌는 구간을 행 순서 기준으로 찾는다. (예: rg3 Injection_Time/Filling_Time 전환점)"""
    v = x.to_numpy()
    change = np.where(v[1:] != v[:-1])[0] + 1
    bounds = np.concatenate(([0], change, [len(v)]))
    segs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        segs.append({"start_pos": int(s), "end_pos": int(e - 1), "value": v[s], "length": int(e - s)})
    return pd.DataFrame(segs)
