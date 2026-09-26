"""주제 ① 사출성형 데이터 — 비가동 블록 정밀 정의, 이상치 유형 규칙.

02_diagnosis_deep.ipynb 의 A-2 ②⑧ 항목에서 사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_quality as dq

LABEL = dq.LABEL
IDLE_MARKER_COLS = dq.IDLE_MARKER_COLS

# cn7 충전 계열 "동시 극단 = 공정 이상 후보" 규칙에 쓰는 변수군 (01 리포트 §3-⑤ 근거)
PROCESS_COLS = ["Injection_Time", "Filling_Time", "Max_Injection_Pressure", "Max_Injection_Speed"]


# ---------------------------------------------------------------------------
# ② 비가동 블록 정밀 정의
# ---------------------------------------------------------------------------

def marker_mode_is_min(U: pd.DataFrame, cols: list[str] = IDLE_MARKER_COLS) -> pd.DataFrame:
    """6개 마커 변수의 고정값(최빈값)이 해당 파일의 최솟값과 같은지 확인한다."""
    rows = []
    for c in cols:
        mode_v = float(U[c].mode().iloc[0])
        min_v = float(U[c].min())
        rows.append({"var": c, "mode": mode_v, "min": min_v, "mode_eq_min": bool(np.isclose(mode_v, min_v))})
    return pd.DataFrame(rows).set_index("var")


def marker_agreement_count(df: pd.DataFrame, cols: list[str] = IDLE_MARKER_COLS) -> pd.Series:
    """각 행에서 6개 마커 변수 중 (해당 파일 자체의) 최빈값과 일치하는 개수."""
    cnt = pd.Series(0, index=df.index)
    for c in cols:
        mode_v = df[c].mode().iloc[0]
        cnt = cnt + (df[c] == mode_v).astype(int)
    return cnt


def idle_sensitivity_table(df: pd.DataFrame, cols: list[str] = IDLE_MARKER_COLS) -> pd.DataFrame:
    """마커 4/5/6개 일치 기준별 비율표."""
    cnt = marker_agreement_count(df, cols)
    n = len(df)
    rows = []
    for k in [4, 5, 6]:
        n_rows = int((cnt >= k).sum())
        rows.append({"min_markers": k, "n_rows": n_rows, "share": n_rows / n if n else np.nan})
    return pd.DataFrame(rows)


def idle_rows_other_var_profile(U: pd.DataFrame, cols: list[str] = IDLE_MARKER_COLS) -> pd.DataFrame:
    """비가동 행(6개 마커 전부 고정)에서 나머지 18개 변수가 어떤 분포를 보이는지 확인한다."""
    mask = dq.idle_block_mask(U)
    other_cols = [c for c in U.columns if c not in cols and c != LABEL]
    idle_stat = U.loc[mask, other_cols].agg(["mean", "std", "min", "max", "nunique"]).T
    full_stat = U.loc[:, other_cols].agg(["mean", "std"]).T.rename(columns={"mean": "mean_all", "std": "std_all"})
    return idle_stat.join(full_stat)


# ---------------------------------------------------------------------------
# ⑧ 이상치 유형 규칙
# ---------------------------------------------------------------------------

def outlier_type(row: pd.Series, process_cols: list[str] = PROCESS_COLS, is_idle: bool = False,
                  z_process: float = 3.0, z_log: float = 5.0, min_process_hits: int = 2) -> str:
    """단일 행을 idle / process / log / normal 로 분류한다 (행은 z-score 표준화된 값).

    우선순위: idle(비가동) > process(공정 이상 후보: process_cols 중 min_process_hits개 이상 동시 극단)
    > log(단독 변수 극단, |z|>z_log) > normal.
    """
    if is_idle:
        return "idle"
    vals = row.drop(labels=[LABEL], errors="ignore")
    n_ext_process = int((vals[process_cols].abs() > z_process).sum())
    if n_ext_process >= min_process_hits:
        return "process"
    if int((vals.abs() > z_log).sum()) >= 1:
        return "log"
    return "normal"


def classify_outliers(df: pd.DataFrame, process_cols: list[str] = PROCESS_COLS, idle_mask: pd.Series | None = None,
                       z_process: float = 3.0, z_log: float = 5.0, min_process_hits: int = 2) -> pd.Series:
    """전체 데이터프레임에 outlier_type 규칙을 벡터화 적용한다 (대량 데이터용)."""
    X = df.drop(columns=[LABEL], errors="ignore")
    if idle_mask is None:
        idle_mask = pd.Series(False, index=df.index)
    n_ext_process = (X[process_cols].abs() > z_process).sum(axis=1)
    n_ext_any = (X.abs() > z_log).sum(axis=1)
    is_process = n_ext_process >= min_process_hits
    is_log = (~is_process) & (n_ext_any >= 1)
    cat = np.select(
        [idle_mask.to_numpy(), is_process.to_numpy(), is_log.to_numpy()],
        ["idle", "process", "log"],
        default="normal",
    )
    return pd.Series(cat, index=df.index, name="outlier_type")


def rule_precision_recall(df: pd.DataFrame, outlier_types: pd.Series, positive_type: str = "process") -> dict:
    """규칙 기반(outlier_type==positive_type) 불량 예측의 정밀도·재현율."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    y = df[LABEL]
    pred = (outlier_types == positive_type).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "n_pred_pos": int(pred.sum()),
        "n_true_pos": int(y.sum()),
        "n_correct": int(((pred == 1) & (y == 1)).sum()),
    }
