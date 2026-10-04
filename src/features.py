"""주제 ③ 프레스 유압펌프 — 윈도우 단위 피처 추출 (진폭 · 형상 · 전류 사인 잔차).

입력은 `preprocess.preprocess()` 결과(샘플 단위, `seg_uid`·`src`·`ts` 포함, 세그먼트 평균 제거·형식 통일 완료)다.
윈도우는 세그먼트 경계 안에서만 만들며(burst 사이 gap을 넘지 않음), 길이가 윈도우보다 짧은 세그먼트는
윈도우가 0개라 "판정 불가"다(`segment_coverage`로 비율을 함께 보고할 것).

피처군 (reports/03_diagnosis_recheck_CHS.md §4)
- shape: 형상 — 윈도우 내 z-score 후 kurt · crest · skew (3채널 × 3 = 9열). 진동 게인 동일성 미확인(A-2) 대비
- amp  : 진폭 — rms · peak · p2p(피크투피크) · kurt · crest · skew (3채널 × 6 = 18열)
- sine : 전류 사인 잔차 — `cur_fit_resid_rms`, `cur_fit_r2`, `cur_fit_f_dev`(|f-0.6|). **세그먼트 단위로 1회 피팅**한
         값을 그 세그먼트의 모든 윈도우에 브로드캐스트한다(윈도우별 재피팅은 느리고 10~30샘플 창에서는
         주파수가 불안정). 따라서 sine 피처는 윈도우 내용이 아니라 세그먼트 전체의 성질이다.
- rel  : 채널 관계 — `vib_corr01`(세그먼트 내 AI0·AI1 피어슨 상관, 02 §6), `cur_ac1`(전류 lag-1 자기상관).
         reference EDA가 강조한 지표(정상 +0.28 → 이상 −0.35, 자기상관 0.93 → 0.44)와 비교하기 위한 것으로,
         sine과 같이 세그먼트 단위로 계산해 브로드캐스트한다. 0.6 Hz 정현파의 lag-1 자기상관은 cos(2π·0.06)≈0.93이라
         `cur_ac1`은 사인 잔차와 정보가 겹친다.
         단 `vib_rms_ratio`(AI0 rms / AI1 rms)는 다른 rel 피처와 달리 세그먼트 브로드캐스트가 아니라 **윈도우 단위**로
         계산한다(AI1 rms가 0이면 nan). 근거: 정상은 하부(AI1)가 더 크고 이상은 상부(AI0)가 역전되는 방향 정보이며
         세그먼트 단독 AUC 0.776 (파인블랭킹 문서 §5).
- relwin: 윈도우 단위 채널 관계(옵트인, 기본 출력에는 포함되지 않음) — `vib_corr01_win`·`cur_ac1_win`(그 윈도우 안
         샘플만으로 계산), `vib_corr01_cum`·`cur_ac1_cum`(세그먼트 시작부터 현재 윈도우 끝까지 누적한 causal 버전).
         rel의 `vib_corr01`·`cur_ac1`은 세그먼트 전체에서 한 번 계산해 모든 윈도우에 브로드캐스트하므로, 세그먼트 초반
         윈도우를 판정할 때 후반 샘플까지 쓰게 되어 실시간 성능으로는 과대평가다. relwin은 이 미래 정보 사용을 없앤 대안이다.
         1초 윈도우(10샘플)의 상관은 분산이 커서 `_win`은 불안정하고, 표본이 누적되는 `_cum`이 실무 후보다.
         `groups`에 "relwin"을 명시한 경우에만 열이 추가되며 기본 `window_features` 출력은 바뀌지 않는다.

`t_abs`(윈도우 시작의 절대 시각)는 시간 블록 분할·정렬용이다. 날짜가 곧 라벨이므로 **모델 입력 금지**.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

import signal_checks as sc
from preprocess import CUR_CHANNEL, SENSORS, add_seg_uid

WINDOWS_S = (1.0, 2.0, 3.0)      # 02 §7: 1~3초. 5초는 선택 효과(이상 세그먼트 5개만 참여)로 폐기
STEP_S = 0.5                     # 윈도우 이동 간격
FS = 10.0
CUR_ALIAS_HZ = 0.6               # 60 Hz가 fs=10 Hz로 접힌 관측 주파수

GROUPS = ("amp", "shape", "sine", "rel")
AMP_KEYS = ("rms", "peak", "p2p", "kurt", "crest", "skew")
SHAPE_KEYS = ("kurt", "crest", "skew")
SINE_KEYS = ("cur_fit_resid_rms", "cur_fit_r2", "cur_fit_f_dev")
REL_KEYS = ("vib_corr01", "cur_ac1", "vib_rms_ratio")
RELWIN_KEYS = ("vib_corr01_win", "cur_ac1_win", "vib_corr01_cum", "cur_ac1_cum")
ALL_GROUPS = GROUPS + ("relwin",)

META_COLS = {"seg_uid", "src", "label", "win_s", "t_start", "t_abs", "n_samples", "fold", "block",
             "state", "vib_grade", "cur_grade"}


def _n(win_s: float, fs: float) -> int:
    return int(round(win_s * fs))


def sliding_windows(seg: pd.DataFrame, win_s: float, step_s: float = STEP_S,
                    fs: float = FS) -> Iterator[tuple[int, pd.DataFrame]]:
    """세그먼트 하나(행 순서 = 시간순)에서 윈도우를 yield 한다: `(시작 샘플 오프셋, 윈도우 프레임)`.

    세그먼트 경계 안에서만 만들고, 길이 < win이면 아무것도 yield 하지 않는다.
    """
    win, step = _n(win_s, fs), max(1, _n(step_s, fs))
    for s in range(0, len(seg) - win + 1, step):
        yield s, seg.iloc[s:s + win]


def amplitude_features(x: np.ndarray) -> dict:
    """진폭 피처: rms, peak(|x| 최댓값), p2p(max−min), kurt(Fisher 초과첨도), crest(peak/rms), skew. x는 DC 제거된 신호."""
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    p2p = float(x.max() - x.min())
    m = x - x.mean()
    m2 = float(np.mean(m ** 2))
    if m2 > 0:
        kurt = float(np.mean(m ** 4) / m2 ** 2 - 3.0)
        skew = float(np.mean(m ** 3) / m2 ** 1.5)
    else:
        kurt = skew = np.nan
    return {"rms": rms, "peak": peak, "p2p": p2p, "kurt": kurt, "crest": peak / rms if rms > 0 else np.nan, "skew": skew}


def shape_features(x: np.ndarray) -> dict:
    """형상 피처: 윈도우 내 z-score((x-평균)/표준편차) 후의 kurt · crest · skew (스케일 무관, rms는 1이라 제외)."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if not sd > 0:
        return {k: np.nan for k in SHAPE_KEYS}
    a = amplitude_features((x - x.mean()) / sd)
    return {k: a[k] for k in SHAPE_KEYS}


def current_sine_features(x: np.ndarray, fs: float = FS) -> dict:
    """전류 사인 피팅 이탈 피처 (`signal_checks.sine_fit_segment` 재사용).

    x는 보통 **세그먼트 전체** 전류다(세그먼트당 1회 계산 후 윈도우에 브로드캐스트).
    cur_fit_resid_rms: 피팅 잔차 RMS / cur_fit_r2: 결정계수 / cur_fit_f_dev: |추정 f - 0.6 Hz|.
    """
    r = sc.sine_fit_segment(np.asarray(x, dtype=float), fs=fs)
    f = r["f_hz"]
    return {"cur_fit_resid_rms": r["resid_rms"], "cur_fit_r2": r["r2"],
            "cur_fit_f_dev": abs(f - CUR_ALIAS_HZ) if f == f else np.nan}


def relation_features(seg: pd.DataFrame) -> dict:
    """채널 관계 피처(세그먼트 단위): vib_corr01 = AI0·AI1 피어슨 상관, cur_ac1 = 전류 lag-1 자기상관. 길이 < 4면 nan.

    윈도우 단위인 `vib_rms_ratio`는 여기 포함하지 않고 `window_features`에서 계산한다.
    """
    if len(seg) < 4:
        return {k: np.nan for k in REL_KEYS if k != "vib_rms_ratio"}
    a, b = seg[SENSORS[0]].to_numpy(dtype=float), seg[SENSORS[1]].to_numpy(dtype=float)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    ac = sc.autocorr(seg[CUR_CHANNEL].to_numpy(dtype=float), max_lag=1)
    return {"vib_corr01": corr, "cur_ac1": float(ac[1]) if len(ac) > 1 else np.nan}


def _corr_ac1(x0: np.ndarray, x1: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """(AI0·AI1 피어슨 상관, 전류 lag-1 자기상관). 길이 < 4면 둘 다 nan, 표준편차 0인 쪽은 nan."""
    if len(x0) < 4:
        return np.nan, np.nan
    x0, x1, cur = (np.asarray(v, dtype=float) for v in (x0, x1, cur))
    corr = float(np.corrcoef(x0, x1)[0, 1]) if x0.std() > 0 and x1.std() > 0 else np.nan
    ac = float(sc.autocorr(cur, max_lag=1)[1]) if cur.std() > 0 else np.nan
    return corr, ac


def window_relation_features(x0: np.ndarray, x1: np.ndarray, cur: np.ndarray) -> dict:
    """윈도우 안 샘플만으로 계산한 관계 피처: vib_corr01_win(AI0·AI1 피어슨 상관), cur_ac1_win(전류 lag-1 자기상관).

    길이 < 4이거나 표준편차가 0이면 nan. `relation_features`와 같은 수식을 윈도우 샘플에만 적용한다.
    """
    corr, ac = _corr_ac1(x0, x1, cur)
    return {"vib_corr01_win": corr, "cur_ac1_win": ac}


def cumulative_relation_features(x0: np.ndarray, x1: np.ndarray, cur: np.ndarray, end: int) -> dict:
    """세그먼트 시작(0)부터 end(배타적, = 윈도우 끝)까지 누적한 causal 관계 피처: vib_corr01_cum, cur_ac1_cum.

    윈도우 끝 이후 샘플을 쓰지 않으므로 실시간 판정에 쓸 수 있다. end가 세그먼트 끝이면 `relation_features` 값과 같다.
    길이 < 4면 nan.
    """
    corr, ac = _corr_ac1(x0[:end], x1[:end], cur[:end])
    return {"vib_corr01_cum": corr, "cur_ac1_cum": ac}


def _label(g: pd.DataFrame) -> int:
    if "Equipment_state" in g.columns:
        return int(round(g["Equipment_state"].mean()))
    return int((g["src"] == "outlier").iloc[0])


def window_features(df_pre: pd.DataFrame, windows_s=WINDOWS_S, step_s: float = STEP_S,
                    groups=GROUPS, fs: float = FS) -> pd.DataFrame:
    """전처리된 샘플 표 → 윈도우 1행짜리 wide 피처 표.

    출력 컬럼: seg_uid, src, label, win_s, t_start(세그먼트 내 오프셋 s), t_abs(윈도우 시작 ts),
    n_samples, 이어서 `<채널>_<피처>`(amp 18), `<채널>_shape_<피처>`(shape 9), cur_fit_*(sine 3), vib_corr01·cur_ac1·vib_rms_ratio(rel 3),
    `groups`에 "relwin"을 넣은 경우에만 마지막에 RELWIN_KEYS 4열(vib_corr01_win·cur_ac1_win·vib_corr01_cum·cur_ac1_cum, 옵트인).
    `t_abs`는 시간 블록 분할용이며 모델 입력이 아니다.
    """
    d = add_seg_uid(df_pre)
    rows = []
    for uid, seg in d.groupby("seg_uid", sort=False):
        src, label = seg["src"].iloc[0], _label(seg)
        sine = current_sine_features(seg[CUR_CHANNEL].to_numpy(), fs) if "sine" in groups else {}
        if "rel" in groups:
            sine = {**sine, **relation_features(seg)}
        arrs = {ch: seg[ch].to_numpy(dtype=float) for ch in SENSORS}
        ts = seg["ts"].to_numpy() if "ts" in seg.columns else None
        for win_s in windows_s:
            win, step = _n(win_s, fs), max(1, _n(step_s, fs))
            for s in range(0, len(seg) - win + 1, step):
                row = {"seg_uid": uid, "src": src, "label": label, "win_s": float(win_s),
                       "t_start": s / fs, "t_abs": ts[s] if ts is not None else pd.NaT, "n_samples": win}
                for ch in SENSORS:
                    x = arrs[ch][s:s + win]
                    if "amp" in groups:
                        row.update({f"{ch}_{k}": v for k, v in amplitude_features(x).items()})
                    if "shape" in groups:
                        row.update({f"{ch}_shape_{k}": v for k, v in shape_features(x).items()})
                row.update(sine)
                if "rel" in groups:
                    r0 = row[f"{SENSORS[0]}_rms"] if "amp" in groups else float(np.sqrt(np.mean(arrs[SENSORS[0]][s:s + win] ** 2)))
                    r1 = row[f"{SENSORS[1]}_rms"] if "amp" in groups else float(np.sqrt(np.mean(arrs[SENSORS[1]][s:s + win] ** 2)))
                    row["vib_rms_ratio"] = r0 / r1 if r1 > 0 else np.nan
                if "relwin" in groups:
                    a0, a1, ac_ = arrs[SENSORS[0]], arrs[SENSORS[1]], arrs[CUR_CHANNEL]
                    row.update(window_relation_features(a0[s:s + win], a1[s:s + win], ac_[s:s + win]))
                    row.update(cumulative_relation_features(a0, a1, ac_, s + win))
                rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """`window_features` 결과에서 **모델 입력 가능한 피처 열**만 고른다(메타·분할 열 제외)."""
    return [c for c in df.columns if c not in META_COLS]


def segment_coverage(df_pre: pd.DataFrame, windows_s=WINDOWS_S, step_s: float = STEP_S,
                     fs: float = FS) -> pd.DataFrame:
    """윈도우 길이별·라벨별(`src`) 세그먼트/윈도우 수와 판정 불가(길이 < win) 세그먼트 비율.

    컬럼: win_s, src, n_seg, n_seg_valid, n_seg_undetermined, undetermined_ratio, n_windows.
    """
    d = add_seg_uid(df_pre)
    seg_len = d.groupby("seg_uid", sort=False).agg(src=("src", "first"), n=("src", "size"))
    rows = []
    for win_s in windows_s:
        win, step = _n(win_s, fs), max(1, _n(step_s, fs))
        nwin = np.where(seg_len["n"] >= win, (seg_len["n"] - win) // step + 1, 0)
        t = seg_len.assign(n_windows=nwin, valid=nwin > 0)
        for src, g in t.groupby("src", sort=False):
            n_seg, n_valid = len(g), int(g["valid"].sum())
            rows.append({"win_s": float(win_s), "src": src, "n_seg": n_seg, "n_seg_valid": n_valid,
                         "n_seg_undetermined": n_seg - n_valid,
                         "undetermined_ratio": round((n_seg - n_valid) / n_seg, 4),
                         "n_windows": int(g["n_windows"].sum())})
    return pd.DataFrame(rows)
