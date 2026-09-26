"""주제 ① 사출성형 데이터 — 로딩 및 품질 진단 유틸리티.

노트북(notebooks/01_data_quality.ipynb)과 이후 전처리 스크립트에서 공통으로 사용한다.
원본 CSV는 절대 수정하지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data_raw"
FIG_DIR = ROOT / "figures"

FILES = {
    "labeled_cn7": "moldset_labeled_cn7.csv",
    "labeled_rg3": "moldset_labeled_rg3.csv",
    "unlabeled_cn7": "moldset_unlabeled_cn7.csv",
    "unlabeled_rg3": "moldset_unlabeled_rg3.csv",
}
LABEL = "PassOrFail"

# 변수 그룹 (사출 공정 지식 기반 분류)
VAR_GROUPS = {
    "time": ["Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time", "Clamp_Close_Time"],
    "position": ["Cushion_Position", "Plasticizing_Position", "Clamp_Open_Position"],
    "speed_rpm": ["Max_Injection_Speed", "Max_Screw_RPM", "Average_Screw_RPM"],
    "pressure": ["Max_Injection_Pressure", "Max_Switch_Over_Pressure", "Max_Back_Pressure", "Average_Back_Pressure"],
    "barrel_temp": [f"Barrel_Temperature_{i}" for i in range(1, 7)],
    "other_temp": ["Hopper_Temperature", "Mold_Temperature_3", "Mold_Temperature_4"],
}

# unlabeled 데이터에서 "비가동 블록"을 판별할 때 쓰는 컬럼 (블록 내부에서 고유값 1개)
IDLE_MARKER_COLS = ["Max_Screw_RPM", "Average_Screw_RPM", "Average_Back_Pressure",
                    "Barrel_Temperature_1", "Mold_Temperature_3", "Mold_Temperature_4"]


def detect_encoding(path: Path, n_bytes: int = 4096) -> str:
    b = path.read_bytes()[:n_bytes]
    if b.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        b.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"


def file_inventory() -> pd.DataFrame:
    rows = []
    for key, name in FILES.items():
        p = RAW_DIR / name
        rows.append({"key": key, "file": name, "size_MB": round(p.stat().st_size / 1e6, 2),
                     "encoding": detect_encoding(p)})
    return pd.DataFrame(rows)


def load(key: str) -> pd.DataFrame:
    """첫 컬럼(이름 없음)은 원본 행 인덱스이므로 index로 사용한다."""
    p = RAW_DIR / FILES[key]
    return pd.read_csv(p, index_col=0, encoding=detect_encoding(p))


def load_all() -> dict[str, pd.DataFrame]:
    return {k: load(k) for k in FILES}


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[LABEL]) if LABEL in df else df.copy()
    y = df[LABEL] if LABEL in df else None
    return X, y


def basic_structure(df: pd.DataFrame) -> pd.DataFrame:
    nun = df.nunique()
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nunique": nun,
        "n_missing": df.isna().sum(),
        "min": df.min().round(3),
        "max": df.max().round(3),
        "flag": np.select([nun <= 1, nun <= 5], ["CONSTANT", "NEAR_CONST"], ""),
    })


def row_key(X: pd.DataFrame, ndigits: int = 8) -> pd.Series:
    """피처 벡터를 문자열 키로 만들어 '같은 입력' 그룹을 식별한다."""
    return X.round(ndigits).astype(str).agg("|".join, axis=1)


def duplicate_report(df: pd.DataFrame) -> dict:
    X, y = split_xy(df)
    key = row_key(X)
    out = {"n_rows": len(df), "full_dup_rows": int(df.duplicated().sum()),
           "feature_dup_rows": int(X.duplicated().sum()),
           "n_distinct_X": int(key.nunique()),
           "adjacent_identical_pairs": int((X.shift(-1) == X).all(axis=1).sum())}
    out["group_size_dist"] = key.value_counts().value_counts().sort_index().to_dict()
    if y is not None:
        g = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["size", "sum", "nunique"])
        out["conflict_groups"] = int((g["nunique"] > 1).sum())
        out["fails_in_conflict"] = int(g.loc[g["nunique"] > 1, "sum"].sum())
        out["total_fails"] = int(y.sum())
        out["distinct_fail_X"] = int((g["sum"] > 0).sum())
        out["both_fail_pairs"] = int((g["sum"] == 2).sum())
    return out


def outlier_table(X: pd.DataFrame, y: pd.Series | None = None, z: float = 3.0) -> pd.DataFrame:
    """값이 이미 표준화돼 있으므로 |값|>z 를 그대로 z-score 기준 이상치로 본다."""
    out = X.abs() > z
    t = pd.DataFrame({"n_outlier": out.sum(), "nunique": X.nunique(),
                      "min": X.min().round(2), "max": X.max().round(2)})
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    t["n_iqr_outlier"] = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()
    if y is not None:
        t["fail_in_outlier"] = out[y == 1].sum()
    return t.sort_values("n_outlier", ascending=False)


def univariate_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """누수 점검용: 단일 변수만으로 라벨을 얼마나 분리하는지(0.5 기준 대칭)."""
    from sklearn.metrics import roc_auc_score
    s = {c: roc_auc_score(y, X[c]) for c in X if X[c].nunique() > 1}
    return pd.Series(s).apply(lambda a: max(a, 1 - a)).sort_values(ascending=False)


def high_corr_pairs(X: pd.DataFrame, th: float = 0.9) -> pd.Series:
    c = X.loc[:, X.nunique() > 1].corr().abs()
    c = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1))
    p = c.stack().sort_values(ascending=False)
    return p[p > th]


def idle_block_mask(U: pd.DataFrame) -> pd.Series:
    """unlabeled 데이터의 비가동(고정값) 블록. 마커 컬럼들이 모두 각자의 최빈값과 같은 행."""
    m = pd.Series(True, index=U.index)
    for c in IDLE_MARKER_COLS:
        m &= U[c] == U[c].mode().iloc[0]
    return m


def run_lengths(mask: pd.Series) -> pd.Series:
    r = mask.astype(int)
    runs = (r != r.shift()).cumsum()
    rl = r.groupby(runs).agg(["first", "size"])
    return rl.loc[rl["first"] == 1, "size"]
