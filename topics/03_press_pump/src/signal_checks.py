"""주제 ③ 프레스 유압펌프 — 신호 해상도·포화·에일리어싱 진단 유틸리티."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]


def resolution_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """채널별 고유값 수·최소 간격(양자화 간격 추정)·유일값 비율 → ADC 비트 수 추정 (추정)."""
    rows = []
    for key, d in dfs.items():
        n = len(d)
        for ch in SENSORS:
            x = np.sort(d[ch].unique())
            diffs = np.diff(x)
            diffs = diffs[diffs > 1e-12]
            step = float(diffs.min()) if len(diffs) else np.nan
            rng = float(x.max() - x.min())
            bits_est = float(np.log2(rng / step + 1)) if step and step > 0 else np.nan
            rows.append({"src": key, "channel": ch, "n_rows": n, "n_unique": len(x),
                         "unique_ratio": round(len(x) / n, 4),
                         "min_step": step, "range": round(rng, 4),
                         "bits_est": round(bits_est, 2) if bits_est == bits_est else np.nan})
    return pd.DataFrame(rows)


def clipping_check(dfs: dict[str, pd.DataFrame], channel: str = "AI2_Current",
                    near_frac: float = 0.99) -> pd.DataFrame:
    """|값| 최댓값 부근이 반복되는지(클리핑/포화) 검사."""
    rows = []
    for key, d in dfs.items():
        absx = d[channel].abs()
        vmax = float(absx.max())
        rows.append({
            "src": key, "channel": channel, "max_abs": round(vmax, 3),
            "count_at_max": int((absx == vmax).sum()),
            f"count_ge_{int(near_frac * 100)}pct_of_max": int((absx >= near_frac * vmax).sum()),
            "share_ge_near_max": round(float((absx >= near_frac * vmax).mean()), 4),
        })
    return pd.DataFrame(rows)


def dc_offset_auc(f_normal: pd.DataFrame, f_outlier: pd.DataFrame,
                   cols: tuple[str, ...] = ("AI0_Vibration_dc", "AI1_Vibration_dc", "AI2_Current_dc")) -> pd.DataFrame:
    """세그먼트 DC 오프셋(세그먼트 평균) 컬럼 하나만으로 normal(0)/outlier(1)를 나눌 때의 AUC.

    AUC가 높으면(예: >0.8) 오프셋만으로도 라벨이 갈린다는 뜻이라 센서 재장착·설정 변경 등의
    누수 가능성을 의심해야 하고, 전처리에서 세그먼트 평균(오프셋) 제거를 규칙으로 삼는다.
    """
    rows = []
    y = np.r_[np.zeros(len(f_normal)), np.ones(len(f_outlier))]
    for col in cols:
        x = np.r_[f_normal[col].to_numpy(), f_outlier[col].to_numpy()]
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)
        rows.append({"column": col, "auc": round(float(auc), 4),
                     "leakage_suspect": bool(auc > 0.8)})
    return pd.DataFrame(rows)


def zero_crossing_by_segment(d: pd.DataFrame, channel: str = "AI2_Current", min_len: int = 10) -> pd.Series:
    """세그먼트별 zero-crossing 주기(샘플 수)를 구한다. min_len 미만 세그먼트는 제외."""
    from data_quality import zero_crossing_period
    out = {}
    for seg_id, g in d.groupby("seg"):
        if len(g) < min_len:
            continue
        out[seg_id] = zero_crossing_period(g[channel].to_numpy())
    return pd.Series(out, name=f"zc_period_{channel}")


def alias_candidates(observed_freq_hz: float, fs: float = 10.0,
                      true_freq_range: tuple[float, float] = (55.0, 65.0), step: float = 0.1) -> pd.DataFrame:
    """관측된 겉보기(에일리어싱) 주파수에 가장 잘 맞는 실제 주파수 후보를 true_freq_range에서 훑는다.

    에일리어싱 공식: alias(f) = f를 fs로 접어(fold) [0, fs/2] 구간에 넣은 값.
    n = round(f/fs) 차수(예: 60Hz, fs=10Hz → n=6)도 함께 보여준다.
    """
    freqs = np.arange(true_freq_range[0], true_freq_range[1] + step / 2, step)
    rows = []
    for f in freqs:
        n = round(f / fs)
        raw = abs(f - n * fs)
        alias = raw if raw <= fs / 2 else fs - raw
        rows.append({"f_true_hz": round(float(f), 2), "n": int(n), "alias_freq_hz": round(float(alias), 4),
                     "diff_from_observed": round(abs(alias - observed_freq_hz), 4)})
    return pd.DataFrame(rows).sort_values("diff_from_observed").reset_index(drop=True)


def fft_spectrum(x: np.ndarray, fs: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """DC 성분을 제거한 실수 FFT 진폭 스펙트럼 (freqs, amplitude)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    amp = np.abs(np.fft.rfft(x - np.mean(x))) / n
    return freqs, amp


def autocorr(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """평균 제거 후 정규화된 자기상관 (lag 0..max_lag)."""
    xc = np.asarray(x, dtype=float) - np.mean(x)
    ac = np.correlate(xc, xc, mode="full")[len(xc) - 1:]
    ac = ac[: max_lag + 1]
    return ac / ac[0] if ac[0] != 0 else ac
