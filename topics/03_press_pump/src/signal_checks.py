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
                   channels: tuple[str, ...] = ("AI0_Vibration", "AI1_Vibration", "AI2_Current")) -> pd.DataFrame:
    """세그먼트 DC 오프셋(세그먼트 평균)만으로 normal(0)/outlier(1)를 나눌 때의 AUC.

    **수정 이력**: 이전 버전은 부호 있는 DC 값에 `max(auc, 1-auc)`를 적용했는데, 이는 오프셋이
    +방향으로 튀는 세그먼트와 -방향으로 튀는 세그먼트가 서로 반대 순위로 기여해 상쇄되는 버그였다
    (부호를 무시하고 "치우쳤는지"만 봐야 하는데 부호까지 하나의 축으로 정렬해버린 것).
    이번 버전은 **|DC|(절대값)** 로 AUC를 계산해 상쇄를 없앴고, 채널별 진폭 스케일이 달라 절대값
    하나만으로는 채널 간 비교가 어려운 점을 보완하고자 **|DC|/RMS(그 세그먼트 자체 진폭 대비 오프셋 비율)**
    AUC도 함께 낸다.
    """
    rows = []
    y = np.r_[np.zeros(len(f_normal)), np.ones(len(f_outlier))]
    for ch in channels:
        dc_col, rms_col = f"{ch}_dc", f"{ch}_rms"
        abs_dc_n, abs_dc_o = f_normal[dc_col].abs(), f_outlier[dc_col].abs()
        x_abs = np.r_[abs_dc_n.to_numpy(), abs_dc_o.to_numpy()]
        auc_abs = roc_auc_score(y, x_abs)
        auc_abs = max(auc_abs, 1 - auc_abs)

        ratio_n = (abs_dc_n / f_normal[rms_col]).replace([np.inf, -np.inf], np.nan)
        ratio_o = (abs_dc_o / f_outlier[rms_col]).replace([np.inf, -np.inf], np.nan)
        y2 = np.r_[np.zeros(ratio_n.notna().sum()), np.ones(ratio_o.notna().sum())]
        x2 = np.r_[ratio_n.dropna().to_numpy(), ratio_o.dropna().to_numpy()]
        auc_ratio = roc_auc_score(y2, x2)
        auc_ratio = max(auc_ratio, 1 - auc_ratio)

        rows.append({"channel": ch, "abs_dc_auc": round(float(auc_abs), 4),
                     "abs_dc_over_rms_auc": round(float(auc_ratio), 4),
                     "leakage_suspect": bool(auc_abs > 0.8)})
    return pd.DataFrame(rows)


def quantization_multiple_share(dfs: dict[str, pd.DataFrame], channel: str = "AI2_Current",
                                 step: float = 1.19209, tol: float = 1e-3) -> pd.DataFrame:
    """채널 값이 step의 정수배에 얼마나 가까운지(공유 비율)를 파일별로 비교한다.

    outlier 파일에서만 비율이 높으면, 두 파일이 서로 다른 계측/전처리 파이프라인(예: 다른 스케일
    팩터로 정수 ADC 카운트를 물리 단위로 환산)을 거쳤다는 뜻이라 "파일 형식 자체가 라벨을 식별하는
    채널"이 될 수 있다(누수).
    """
    rows = []
    for key, d in dfs.items():
        x = d[channel].to_numpy()
        q = x / step
        frac = np.abs(q - np.round(q))
        share = float((frac < tol).mean())
        rows.append({"src": key, "channel": channel, "step": step, "tol": tol,
                     "share_multiple_of_step": round(share, 4)})
    return pd.DataFrame(rows)


def decimal_places_table(dfs: dict[str, pd.DataFrame],
                          channels: tuple[str, ...] = ("AI0_Vibration", "AI1_Vibration")) -> pd.DataFrame:
    """값의 소수점 이하 자릿수 분포(최빈값과 그 비중)를 파일별로 비교한다.

    같은 물리량이라도 내보내기(export) 방식이 다르면 소수 자릿수 포맷이 달라질 수 있다 —
    이 차이가 파일(=라벨)마다 체계적으로 다르면 값 자체가 아니라 "표기 형식"만으로도 라벨을
    맞출 수 있다는 뜻이라 심각한 누수다.
    """
    rows = []
    for key, d in dfs.items():
        for ch in channels:
            ndec = d[ch].astype(str).str.split(".").str[1].str.len()
            mode = int(ndec.mode().iloc[0])
            share_mode = float((ndec == mode).mean())
            rows.append({"src": key, "channel": ch, "mode_decimals": mode,
                         "share_at_mode": round(share_mode, 4),
                         "decimal_range": f"{int(ndec.min())}~{int(ndec.max())}"})
    return pd.DataFrame(rows)


def zero_crossing_demeaned_by_segment(d: pd.DataFrame, channel: str = "AI2_Current",
                                       min_len: int = 10) -> pd.Series:
    """세그먼트 평균(DC)을 뺀 뒤의 zero-crossing 주기. 큰 DC 편이가 zero-crossing을 없애는지 확인용."""
    from data_quality import zero_crossing_period
    out = {}
    for seg_id, g in d.groupby("seg"):
        if len(g) < min_len:
            continue
        x = g[channel].to_numpy()
        out[seg_id] = zero_crossing_period(x - x.mean())
    return pd.Series(out, name=f"zc_demeaned_{channel}")


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
