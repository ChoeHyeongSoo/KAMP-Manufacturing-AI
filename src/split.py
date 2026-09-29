"""주제 ③ 프레스 유압펌프 — 누수 방지 분할 (세그먼트 GroupKFold · 정상 시간순 블록).

02 진단 §9의 분할 규칙을 코드로 고정한다. 모든 모델 노트북(2x)은 이 모듈의 분할만 쓴다 —
분할이 노트북마다 다르면 `results/model_comparison.csv` 비교가 성립하지 않는다.

- 왜 세그먼트 단위인가: 같은 burst(세그먼트) 안의 샘플·윈도우는 서로 거의 같은 값이라 샘플 단위로
  나누면 train/test에 같은 세그먼트가 걸쳐 성능이 부풀려진다(그룹 누수). 그래서 분할 키는 항상 `seg_uid`.
- 왜 시간 블록도 보는가: 정상 파일은 77분 한 날짜, 이상 파일은 다른 날짜 하나뿐이라 "날짜 = 라벨"이다.
  정상 안에서 시간 추세(드리프트)가 있으면 앞 시간대로 잡은 임계값이 뒤 시간대에서 오경보를 늘리므로,
  정상 세그먼트를 시간순 블록으로 나눠 forward-chaining으로도 평가한다(02 §7 fpr 표와 연결).

두 함수 모두 sklearn 스타일로 `(train_idx, test_idx)` 위치 인덱스 쌍을 yield 한다. 입력 표는
세그먼트 단위(`segments.csv`)여도 되고, 윈도우/샘플 단위(`seg_uid` 컬럼 포함)여도 된다 — 어느 경우든
분할은 `seg_uid` 기준으로 이뤄지므로 같은 세그먼트가 train/test에 동시에 들어가지 않는다.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

SEG_KEY = "seg_uid"
LABEL_COL = "label"      # 세그먼트 라벨 (0 정상, 1 이상) — segments.csv 기준
START_COL = "start"      # 세그먼트 시작 시각


def _check_cols(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"필요 컬럼 없음: {missing} (segments.csv 또는 seg_uid를 붙인 표를 넣을 것)")


def _seg_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    """seg_uid → 세그먼트 라벨(0/1). 샘플 단위 표면 세그먼트 평균을 반올림한다."""
    return df.groupby(SEG_KEY)[label_col].mean().round().astype(int)


# --- 1) 세그먼트 GroupKFold (계층화) ---------------------------------------

def fold_assignment(df: pd.DataFrame, n_splits: int = 5, seed: int = 0,
                    label_col: str = LABEL_COL) -> pd.Series:
    """seg_uid → fold 번호(0..n_splits-1).

    라벨별로 세그먼트를 섞은 뒤 round-robin으로 배정하므로 각 fold의 이상 세그먼트 수가 최대 1개
    차이(21개 → 4·4·4·4·5)로 고르다. sklearn StratifiedGroupKFold와 달리 결과가 seed에 대해 결정적이고
    이상 세그먼트가 한 fold에 몰리는 일이 없다.
    """
    _check_cols(df, SEG_KEY, label_col)
    seg_lab = _seg_label(df, label_col)
    rng = np.random.default_rng(seed)
    assign: dict[str, int] = {}
    for lab in sorted(seg_lab.unique()):
        uids = seg_lab.index[seg_lab == lab].to_numpy()
        rng.shuffle(uids)
        for i, uid in enumerate(uids):
            assign[uid] = i % n_splits
    return pd.Series(assign, name="fold").reindex(seg_lab.index)


def group_kfold_seg(df: pd.DataFrame, n_splits: int = 5, seed: int = 0,
                    label_col: str = LABEL_COL, min_pos_per_fold: int = 4
                    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """세그먼트(`seg_uid`) 단위 계층화 K-fold. `(train_idx, test_idx)` 위치 인덱스를 yield 한다.

    각 test fold에 이상 세그먼트가 `min_pos_per_fold` 미만이면 ValueError — 이상 21개, 5-fold 기준 4개
    보장. n_splits를 늘려 조건이 깨지면 min_pos_per_fold를 낮춰 명시적으로 허용한다.
    """
    folds = fold_assignment(df, n_splits, seed, label_col)
    seg_lab = _seg_label(df, label_col)
    for k in range(n_splits):
        n_pos = int(((folds == k) & (seg_lab == 1)).sum())
        if n_pos < min_pos_per_fold:
            raise ValueError(f"fold {k}의 이상 세그먼트 {n_pos}개 < {min_pos_per_fold} (n_splits={n_splits})")
    row_fold = df[SEG_KEY].map(folds).to_numpy()
    idx = np.arange(len(df))
    for k in range(n_splits):
        yield idx[row_fold != k], idx[row_fold == k]


# --- 2) 정상 세그먼트 시간순 블록 (forward-chaining) ----------------------

def block_assignment(df: pd.DataFrame, n_blocks: int = 5, start_col: str = START_COL) -> pd.Series:
    """seg_uid → 시간순 블록 번호(0..n_blocks-1). 세그먼트 시작 시각 순으로 균등 개수 분할."""
    _check_cols(df, SEG_KEY, start_col)
    starts = df.groupby(SEG_KEY)[start_col].min().sort_values()
    blocks = np.floor(np.arange(len(starts)) * n_blocks / len(starts)).astype(int)
    return pd.Series(blocks, index=starts.index, name="block")


def time_blocks(df: pd.DataFrame, n_blocks: int = 5, start_col: str = START_COL,
                min_train_blocks: int = 1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """정상 세그먼트를 시간순 n블록으로 나눠 forward-chaining `(train_idx, test_idx)`를 yield 한다.

    k번째 반복: train = 블록 0..k-1, test = 블록 k (k = min_train_blocks .. n_blocks-1).
    정상 파일만 넣는 것이 원칙이다(이상 파일은 다른 날짜라 시간순 배치가 무의미). 이상 세그먼트를 함께
    평가하려면 호출 측에서 test에 이상 세그먼트를 별도로 합친다(`append_outliers` 참고).
    """
    blocks = block_assignment(df, n_blocks, start_col)
    row_block = df[SEG_KEY].map(blocks).to_numpy()
    idx = np.arange(len(df))
    for k in range(min_train_blocks, n_blocks):
        yield idx[row_block < k], idx[row_block == k]


def append_outliers(test_idx: np.ndarray, df: pd.DataFrame, label_col: str = LABEL_COL) -> np.ndarray:
    """time_blocks의 test 인덱스에 이상(label==1) 행 전체를 덧붙인다 — 정상+이상 혼합 표에서 시간 블록 평가용."""
    _check_cols(df, label_col)
    out_idx = np.flatnonzero(df[label_col].to_numpy() == 1)
    return np.concatenate([test_idx, out_idx])


# --- 3) 요약표 -----------------------------------------------------------

def split_summary(df: pd.DataFrame, assignment: pd.Series, label_col: str = LABEL_COL,
                  state_col: str = "state", start_col: str = START_COL) -> pd.DataFrame:
    """fold/block별 세그먼트 수·이상 수·행 수·(있으면) 상태 비율·시각 범위 표 — 리포트 분할 절용.

    `assignment`는 `fold_assignment` 또는 `block_assignment`의 결과(seg_uid → 번호).
    """
    _check_cols(df, SEG_KEY, label_col)
    seg = df.groupby(SEG_KEY).agg(n_rows=(SEG_KEY, "size"), label=(label_col, "mean"))
    seg["label"] = seg["label"].round().astype(int)
    if start_col in df.columns:
        seg["start"] = df.groupby(SEG_KEY)[start_col].min()
    if state_col in df.columns:
        seg["state"] = df.groupby(SEG_KEY)[state_col].first()
    seg[assignment.name] = assignment.reindex(seg.index)
    g = seg.groupby(assignment.name)
    out = pd.DataFrame({
        "n_segments": g.size(),
        "n_outlier_seg": g["label"].sum(),
        "n_rows": g["n_rows"].sum(),
    })
    if "start" in seg.columns:
        out["start_min"] = g["start"].min()
        out["start_max"] = g["start"].max()
    if "state" in seg.columns:
        st = seg[seg["label"] == 0].groupby(assignment.name)["state"].value_counts(normalize=True).unstack()
        st.columns = [f"share_state{c}" for c in st.columns]
        out = out.join(st.round(4))
    return out
