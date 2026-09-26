"""주제 ③ 프레스 유압펌프 — 세그먼트(burst) 단위 피처·등급·운전 상태.

`data_quality.load`/`segment_table`/`rolling_rms`/`zero_crossing_period`를 재사용해
세그먼트별 피처표를 만들고, 이상 21세그먼트의 3등급과 정상 599세그먼트의 운전 상태를
붙인 `data_processed/segments.csv`를 만드는 함수를 제공한다.
data_processed/는 git 제외 대상이므로 이 파일의 함수로 언제든 재생성한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import data_quality as dq

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT / "data_processed"

VIB_CHANNELS = ["AI0_Vibration", "AI1_Vibration"]
CUR_CHANNEL = "AI2_Current"
STATE_FEATURES = ["AI0_Vibration_rms", "AI1_Vibration_rms", "AI2_Current_rms", "AI2_Current_peak"]


def _crest(x: np.ndarray, rms: float) -> float:
    return float(np.max(np.abs(x)) / rms) if rms > 0 else np.nan


def segment_features(d: pd.DataFrame) -> pd.DataFrame:
    """`dq.load(key)`의 결과(단일 파일, `seg` 컬럼 포함)에서 세그먼트별 피처표를 만든다.

    진동 AI0/AI1: RMS·피크·첨도(kurtosis)·crest factor·DC 오프셋(세그먼트 평균).
    전류 AI2: RMS·피크·DC 오프셋. 길이(len)·시작 시각(start)·라벨 포함.
    """
    rows = []
    for seg_id, g in d.groupby("seg"):
        row = {
            "seg_id": int(seg_id), "src": g["src"].iloc[0],
            "start": g["ts"].min(), "len": len(g),
            "label": float(g[dq.LABEL].mean()),
        }
        for ch in VIB_CHANNELS:
            x = g[ch].to_numpy()
            rms = float(np.sqrt(np.mean(x ** 2)))
            row[f"{ch}_rms"] = rms
            row[f"{ch}_peak"] = float(np.max(np.abs(x)))
            row[f"{ch}_kurt"] = float(stats.kurtosis(x)) if len(x) > 3 else np.nan
            row[f"{ch}_crest"] = _crest(x, rms)
            row[f"{ch}_dc"] = float(np.mean(x))
        x = g[CUR_CHANNEL].to_numpy()
        rms = float(np.sqrt(np.mean(x ** 2)))
        row[f"{CUR_CHANNEL}_rms"] = rms
        row[f"{CUR_CHANNEL}_peak"] = float(np.max(np.abs(x)))
        row[f"{CUR_CHANNEL}_dc"] = float(np.mean(x))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("seg_id")
    out["vib_rms"] = np.sqrt(out["AI0_Vibration_rms"] ** 2 + out["AI1_Vibration_rms"] ** 2)
    for ch in VIB_CHANNELS + [CUR_CHANNEL]:
        out[f"{ch}_dc_abs"] = out[f"{ch}_dc"].abs()
    return out


def grade_outlier_segments(f_outlier: pd.DataFrame, f_normal: pd.DataFrame,
                            col: str = "vib_rms", q_low: float = 0.50, q_high: float = 0.99) -> pd.Series:
    """이상 세그먼트를 정상 세그먼트 `col` 분위수 기준 3등급으로 나눈다.

    규칙: col > 정상 q_high(기본 99%ile) → "확실 이상"
          col <= 정상 q_low(기본 50%ile, 중앙값) → "정상 유사"
          그 사이 → "경계"
    """
    lo, hi = f_normal[col].quantile([q_low, q_high])

    def _grade(v: float) -> str:
        if v > hi:
            return "확실 이상"
        if v <= lo:
            return "정상 유사"
        return "경계"

    return f_outlier[col].apply(_grade)


def choose_k_operating_states(f_normal: pd.DataFrame, ks: tuple[int, ...] = (2, 3, 4),
                               features: list[str] | None = None, random_state: int = 0):
    """정상 세그먼트 피처를 표준화 후 k=2~4 k-means를 돌려 실루엣 점수로 k를 고른다.

    반환: (선택된 k, {k: (labels, silhouette)} 전체 결과, 표준화된 X)
    """
    features = features or STATE_FEATURES
    X = StandardScaler().fit_transform(f_normal[features])
    results = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        sil = silhouette_score(X, km.labels_)
        results[k] = (km.labels_, sil)
    best_k = max(results, key=lambda k: results[k][1])
    return best_k, results, X


def operating_state_summary(f_normal: pd.DataFrame, state_col: str = "state",
                             features: list[str] | None = None) -> pd.DataFrame:
    features = features or STATE_FEATURES
    g = f_normal.groupby(state_col)
    summ = g[features].mean().round(3)
    summ["n_segments"] = g.size()
    summ["n_samples"] = g["len"].sum()
    summ["share_samples"] = (summ["n_samples"] / summ["n_samples"].sum()).round(4)
    return summ


def build_segments_table(save: bool = True) -> pd.DataFrame:
    """normal+outlier 세그먼트 피처를 합쳐 등급·운전상태를 붙인 표를 만들고 저장한다.

    산출 컬럼: seg_uid, src, seg_id, start, len, label, (피처...),
    vib_grade/cur_grade(이상만, 진동 기준/전류 기준 각각의 등급), state(정상만)

    진동 기준(vib_grade)과 전류 기준(cur_grade) 등급이 갈리는 세그먼트(예: 19·20 — 진동은
    "정상 유사"인데 전류는 "확실 이상")가 있으므로, 하나의 등급으로 뭉개지 않고 두 열을 모두 남긴다.
    """
    dfs = dq.load_all()
    N, O = dfs["normal"], dfs["outlier"]
    fN = segment_features(N)
    fO = segment_features(O)

    fO["vib_grade"] = grade_outlier_segments(fO, fN, col="vib_rms")
    fO["cur_grade"] = grade_outlier_segments(fO, fN, col="AI2_Current_dc_abs")
    best_k, results, X = choose_k_operating_states(fN)
    fN["state"] = results[best_k][0]

    fN["vib_grade"] = pd.NA
    fN["cur_grade"] = pd.NA
    fO["state"] = pd.NA

    both = pd.concat([fN, fO], axis=0)
    both = both.reset_index().rename(columns={"seg_id": "seg_id"})
    both.insert(0, "seg_uid", both["src"] + "_" + both["seg_id"].astype(str))
    both = both.sort_values(["src", "seg_id"]).reset_index(drop=True)

    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        both.to_csv(DATA_PROCESSED_DIR / "segments.csv", index=False, encoding="utf-8-sig")
    return both


# --- 채널 간 관계 ---------------------------------------------------------

def within_segment_corr(d: pd.DataFrame, ch_a: str = "AI0_Vibration", ch_b: str = "AI1_Vibration",
                         min_len: int = 5) -> pd.Series:
    """세그먼트별로 두 채널의 원시 시계열 상관(피어슨)을 구한다."""
    out = {}
    for seg_id, g in d.groupby("seg"):
        if len(g) < min_len:
            continue
        r = np.corrcoef(g[ch_a], g[ch_b])[0, 1]
        out[seg_id] = r
    return pd.Series(out, name=f"corr_{ch_a}_{ch_b}")


def separability_auc(f_normal: pd.DataFrame, f_outlier: pd.DataFrame, col: str) -> float:
    """세그먼트 피처 컬럼 하나만으로 normal(0)/outlier(1)를 나눴을 때의 AUC (대칭화: >=0.5)."""
    y = np.r_[np.zeros(len(f_normal)), np.ones(len(f_outlier))]
    x = np.r_[f_normal[col].to_numpy(), f_outlier[col].to_numpy()]
    auc = roc_auc_score(y, x)
    return float(max(auc, 1 - auc))


# --- 윈도우 민감도·탐지 지연 ------------------------------------------------

def window_auc_table(d_normal: pd.DataFrame, d_outlier: pd.DataFrame,
                      windows_sec: tuple[float, ...] = (1, 2, 3, 5), fs: float = 10.0,
                      channels: tuple[str, ...] = (VIB_CHANNELS[0], VIB_CHANNELS[1], CUR_CHANNEL)) -> pd.DataFrame:
    """세그먼트 경계를 넘지 않는 이동 RMS(`dq.rolling_rms` 재사용)로 윈도우 길이별 AUC를 구한다.

    **주의(선택 효과)**: 윈도우가 길어질수록 그 윈도우를 채울 수 있는(길이 >= win) 이상 세그먼트 수 자체가
    줄어든다(예: 5초=50샘플 윈도우는 길이 50인 이상 세그먼트 5개만 참여). `n_outlier_segments_qualify`로
    이 효과를 같이 보여주므로, 윈도우 간 AUC를 비교할 때 표본이 달라졌다는 점을 함께 읽어야 한다.
    (동일 세그먼트 집합으로 고정한 공정 비교는 `window_auc_fixed_length` 참고.)
    """
    rows = []
    seg_len_out = d_outlier.groupby("seg").size()
    for sec in windows_sec:
        win = int(round(sec * fs))
        rN = dq.rolling_rms(d_normal, win=win)
        rO = dq.rolling_rms(d_outlier, win=win)
        n_outlier_segments_qualify = int((seg_len_out >= win).sum())
        for ch in channels:
            rn = rN[ch].dropna()
            ro = rO[ch].dropna()
            y = np.r_[np.zeros(len(rn)), np.ones(len(ro))]
            x = np.r_[rn.to_numpy(), ro.to_numpy()]
            auc = roc_auc_score(y, x)
            rows.append({"window_sec": sec, "channel": ch, "auc": round(max(auc, 1 - auc), 4),
                         "n_normal": len(rn), "n_outlier": len(ro),
                         "n_outlier_segments_qualify": n_outlier_segments_qualify})
    return pd.DataFrame(rows)


def window_auc_fixed_length(d_normal: pd.DataFrame, d_outlier: pd.DataFrame, length: int = 50,
                             windows_sec: tuple[float, ...] = (1, 2, 3, 5), fs: float = 10.0,
                             channels: tuple[str, ...] = (VIB_CHANNELS[0], VIB_CHANNELS[1], CUR_CHANNEL)) -> pd.DataFrame:
    """길이가 정확히 `length`인 세그먼트끼리만(정상·이상 동일 집합) 윈도우별 AUC를 비교한다.

    `window_auc_table`은 윈도우가 길어질수록 참여하는 이상 세그먼트 수가 줄어드는 선택 효과가 있어
    윈도우 간 비교가 공정하지 않다. 이 함수는 세그먼트 집합을 고정해 그 효과를 제거한다(다만 세그먼트
    수 자체가 애초에 작으면 - 예: 이상 길이 50 세그먼트 5개 - 5초 지점은 여전히 표본이 작다는 한계가 남는다).
    """
    segN_len = d_normal.groupby("seg").size()
    segO_len = d_outlier.groupby("seg").size()
    segs_n = segN_len[segN_len == length].index
    segs_o = segO_len[segO_len == length].index
    dN = d_normal[d_normal["seg"].isin(segs_n)]
    dO = d_outlier[d_outlier["seg"].isin(segs_o)]
    tab = window_auc_table(dN, dO, windows_sec=windows_sec, fs=fs, channels=channels)
    tab = tab.drop(columns=["n_outlier_segments_qualify"])
    tab["n_normal_segments_fixed"] = len(segs_n)
    tab["n_outlier_segments_fixed"] = len(segs_o)
    return tab


def leakage_free_fpr_table(d_normal: pd.DataFrame, win: int, channel: str,
                            fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8), q: float = 0.99) -> pd.DataFrame:
    """세그먼트 순서(시간순) 앞 f% 세그먼트로 임계(q분위)를 잡고 나머지 정상 세그먼트의 오경보율을 본다.

    오경보율은 두 단위로 함께 낸다: **샘플 단위**(`fpr_sample`, 윈도우 하나하나가 임계 초과했는지)와
    **세그먼트 단위**(`fpr_segment`, 세그먼트 안에 임계 초과 윈도우가 하나라도 있으면 그 세그먼트 전체를
    "오경보 1건"으로 셈). 현장에서는 세그먼트(=한 번의 점검 burst) 단위로 경보가 발생하므로 세그먼트
    단위 오경보율이 실제 운영 지표에 더 가깝다. 세그먼트 단위는 윈도우를 하나도 완성 못하는(길이<win)
    세그먼트를 분모에서 제외한다(`n_test_seg_valid`).
    """
    seg_ids = np.sort(d_normal["seg"].unique())
    n_segs = len(seg_ids)
    rms = dq.rolling_rms(d_normal, win=win)[channel]
    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * n_segs)))
        train_segs, test_segs = set(seg_ids[:k]), set(seg_ids[k:])
        train_mask = d_normal["seg"].isin(train_segs).to_numpy()
        test_mask = d_normal["seg"].isin(test_segs).to_numpy()
        thr = rms[train_mask].dropna().quantile(q)
        test_vals = rms[test_mask].dropna()
        fpr_sample = float((test_vals > thr).mean()) if len(test_vals) else np.nan

        test_df = d_normal.loc[test_mask, ["seg"]].copy()
        test_df["rms"] = rms[test_mask]
        g = test_df.groupby("seg")["rms"]
        n_valid = g.apply(lambda s: s.notna().sum())
        seg_exceed = g.apply(lambda s: bool((s.dropna() > thr).any()))
        valid_segs = n_valid[n_valid > 0].index
        fpr_segment = float(seg_exceed.loc[valid_segs].mean()) if len(valid_segs) else np.nan

        rows.append({"train_frac": frac, "n_train_seg": k, "n_test_seg": n_segs - k,
                     "threshold_q99": round(float(thr), 4), "n_test_samples": len(test_vals),
                     "fpr_sample": round(fpr_sample, 4),
                     "n_test_seg_valid": int(len(valid_segs)), "fpr_segment": round(fpr_segment, 4)})
    return pd.DataFrame(rows)


def detection_delay_table(d_outlier: pd.DataFrame, threshold: float, win: int, channel: str,
                           fs: float = 10.0, quiet_segments: tuple[int, ...] = (3, 19, 20)) -> pd.DataFrame:
    """세그먼트 시작 후 이동 RMS가 threshold를 처음 넘는 샘플 위치(지연)를 구한다.

    세그먼트 길이가 win보다 짧아 창을 한 번도 완성하지 못하는 경우와, 조용한 세그먼트(quiet_segments)로
    실제 미탐지인 경우를 구분해 `reason` 컬럼에 남긴다.
    """
    rms = dq.rolling_rms(d_outlier, win=win)[channel]
    rows = []
    for seg_id, g in d_outlier.groupby("seg"):
        seg_id = int(seg_id)
        r = rms.loc[g.index].reset_index(drop=True)
        n_valid = r.notna().sum()
        exceed = r[r > threshold]
        if len(exceed):
            delay = int(exceed.index[0])
            detected, reason = True, "탐지"
        else:
            delay = np.nan
            if n_valid == 0:
                detected, reason = False, "세그먼트 길이<윈도우(판정 불가)"
            elif seg_id in quiet_segments:
                detected, reason = False, "조용한 세그먼트(라벨 노이즈 의심)"
            else:
                detected, reason = False, "미탐지"
        rows.append({"seg": seg_id, "n": len(g), "n_valid_windows": int(n_valid),
                     "detected": detected, "delay_samples": delay,
                     "delay_sec": round(delay / fs, 2) if delay == delay else np.nan,
                     "reason": reason})
    return pd.DataFrame(rows)


# --- 이상 구간 시간 진행 -----------------------------------------------------

def outlier_trend(f_outlier: pd.DataFrame, col: str = "vib_rms"):
    """시간순 정렬된 이상 21세그먼트의 col에 대한 선형 기울기·Spearman 상관을 구한다."""
    fo = f_outlier.sort_values("start")
    t = np.arange(len(fo))
    slope, intercept = np.polyfit(t, fo[col].to_numpy(), 1)
    rho, p = stats.spearmanr(t, fo[col].to_numpy())
    return {"slope": float(slope), "intercept": float(intercept), "spearman_rho": float(rho), "spearman_p": float(p)}
