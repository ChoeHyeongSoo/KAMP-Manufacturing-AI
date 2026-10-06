"""주제 ③ — JIW 14 burst 사이 공백 보간 정합성 검증.

질문: burst 사이 공백(측정되지 않은 구간)을 보간해서 연속 신호로 모델에 넘겨도 되는가? (docs/final_design_JIW.md §2.1~2.2)
공백의 실제 값은 없으므로, **값이 있는 구간을 가려서** 보간이 복원하는지로 가능 범위를 잰다.

- 검증 A (burst 안 마스킹): 정상 burst 가운데 k샘플을 가리고 선형·3차 스플라인·사인 피팅·0 채움으로 복원한다.
- 검증 B (실제 공백 홀드아웃): 정상 burst를 통째로 가리고 앞뒤 burst(실제 공백 포함)만으로 복원한다. 가장 실제에 가까운 검증이다.
- 판정: 가린 구간의 **채널별 합산 R²**(분산 설명 비율) ≥ 0.9이면 그 조건에서 보간 가능.

입력은 표준 전처리(`preprocess.preprocess()`) 결과다. 세그먼트마다 평균이 제거돼 있어 burst 사이 DC는 이어지지 않으므로,
사인 모델은 burst마다 오프셋을 따로 두고(공통 오프셋 금지) 파형(진폭·주파수·위상)의 연속성만 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import paths
import preprocess

FS = 10.0
CH = preprocess.SENSORS
F_GRID = np.linspace(0.3, 1.2, 181)      # 정상 전류는 0.6 Hz로 접힌 사인(03 A-1)이라 그 주변만 탐색


def load_normal() -> tuple[dict, pd.DataFrame]:
    """정상 세그먼트 배열(표준 전처리)과 시간순 표(seg_uid, start_s, len). start_s = 첫 정상 burst 시작 기준 초."""
    import benchmark_jiw as B
    segs, _ = B.load_inputs()
    S = pd.read_csv(paths.DATA_PROCESSED / "segments.csv", encoding="utf-8-sig")
    S = S[S["label"] == 0].copy()
    S["start_s"] = (pd.to_datetime(S["start"]) - pd.to_datetime(S["start"]).min()).dt.total_seconds()
    S = S.sort_values("start_s").reset_index(drop=True)
    return {u: segs[u] for u in S["seg_uid"]}, S[["seg_uid", "start_s", "len"]]


def real_gaps(S: pd.DataFrame) -> pd.Series:
    """연속한 정상 burst 사이 실제 공백(초) = 다음 시작 − (이번 시작 + 길이 × 0.1)"""
    end = S["start_s"] + S["len"] / FS
    return (S["start_s"].shift(-1) - end).dropna().reset_index(drop=True)


# ------------------------------------------------------------------ 보간기
def _sine_design(t, f, seg_id, n_seg):
    cols = [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)] + [(seg_id == i).astype(float) for i in range(n_seg)]
    return np.column_stack(cols)


def sine_fit_predict(t_vis, y_vis, seg_vis, t_new) -> np.ndarray:
    """보이는 점에 A·sin + B·cos + (세그먼트별 오프셋)을 최소제곱으로 맞추고 t_new에서 **오프셋 없는 파형**을 예측한다.
    주파수는 F_GRID에서 탐색 후 ±0.01 Hz를 다시 훑는다."""
    ids = np.unique(seg_vis)
    sid = np.searchsorted(ids, seg_vis)
    def solve(f):
        M = _sine_design(t_vis, f, sid, len(ids))
        coef, *_ = np.linalg.lstsq(M, y_vis, rcond=None)
        r = y_vis - M @ coef
        return float(r @ r), coef
    best = min(((solve(f)[0], f) for f in F_GRID))
    fine = np.linspace(best[1] - 0.01, best[1] + 0.01, 21)
    f = fine[int(np.argmin([solve(x)[0] for x in fine]))]
    coef = solve(f)[1]
    return coef[0] * np.sin(2 * np.pi * f * t_new) + coef[1] * np.cos(2 * np.pi * f * t_new)


def fill(method: str, t_vis, y_vis, t_new, seg_vis=None) -> np.ndarray:
    """가린 지점 t_new의 값을 보이는 점으로 복원한다. method: zero | linear | cubic | sine"""
    if method == "zero":
        return np.zeros(len(t_new))             # 세그먼트 평균 제거 후 기댓값
    if method == "linear":
        return np.interp(t_new, t_vis, y_vis)
    if method == "cubic":
        return CubicSpline(t_vis, y_vis)(t_new)
    if method == "sine":
        seg = np.zeros(len(t_vis), dtype=int) if seg_vis is None else seg_vis
        return sine_fit_predict(t_vis, y_vis, seg, t_new)
    raise ValueError(method)


METHODS = ["zero", "linear", "cubic", "sine"]
METHOD_KO = {"zero": "0 채움(평균)", "linear": "선형", "cubic": "3차 스플라인", "sine": "사인 피팅"}


def _score(rows: list) -> dict:
    """rows: [(y_true 배열, y_pred 배열)] → 합산 R²·nRMSE 구성요소"""
    y = np.concatenate([a for a, _ in rows])
    p = np.concatenate([b for _, b in rows])
    return {"n": len(y), "sse": float(((y - p) ** 2).sum()), "sst": float(((y - y.mean()) ** 2).sum()), "y_sd": float(y.std())}


# ------------------------------------------------------------------ 검증 A: burst 안 마스킹
def mask_eval(segs: dict, ks=(5, 10, 15, 20, 30), min_ctx: int = 5, methods=METHODS, progress=False) -> pd.DataFrame:
    """정상 burst 가운데 k샘플을 가리고 복원. 앞뒤 각각 min_ctx샘플 이상이 남는 burst만 쓴다."""
    out = []
    for k in ks:
        use = [u for u, x in segs.items() if len(x) >= k + 2 * min_ctx]
        for m in methods:
            for ci, ch in enumerate(CH):
                rows = []
                for u in use:
                    x = segs[u][:, ci].astype(float)
                    L = len(x)
                    s = (L - k) // 2
                    miss = np.arange(s, s + k)
                    vis = np.setdiff1d(np.arange(L), miss)
                    t = np.arange(L) / FS
                    pred = fill(m, t[vis], x[vis], t[miss], np.zeros(len(vis), dtype=int))
                    rows.append((x[miss], pred))
                sc = _score(rows)
                out.append({"k": k, "초": k / FS, "방법": m, "채널": ch, "burst 수": len(use), "표본": sc["n"], "R2": 1 - sc["sse"] / sc["sst"],
                            "nRMSE": (sc["sse"] / sc["n"]) ** 0.5 / sc["y_sd"]})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 검증 B: 실제 공백 홀드아웃
def holdout_eval(segs: dict, S: pd.DataFrame, methods=("zero", "linear", "sine"), min_len: int = 20) -> pd.DataFrame:
    """연속한 세 정상 burst (i−1, i, i+1) 중 가운데 i를 통째로 가리고 앞뒤 burst만으로 복원한다.
    span = i−1 끝 ~ i+1 시작 (= 실제 공백 2개 + 가린 burst 길이). R²는 burst마다 가린 값의 평균을 뺀 파형 기준(평균 제거 데이터와 같은 조건)."""
    rows = []
    uids = list(S["seg_uid"])
    st = S.set_index("seg_uid")["start_s"].to_dict()
    for j in range(1, len(uids) - 1):
        a, b, c = uids[j - 1], uids[j], uids[j + 1]
        if min(len(segs[a]), len(segs[b]), len(segs[c])) < min_len:
            continue
        ta, tb, tc = (st[u] + np.arange(len(segs[u])) / FS for u in (a, b, c))
        t_vis, seg_vis = np.concatenate([ta, tc]), np.r_[np.zeros(len(ta), dtype=int), np.ones(len(tc), dtype=int)]
        span = tc[0] - ta[-1]
        for ci, ch in enumerate(CH):
            y_vis = np.concatenate([segs[a][:, ci], segs[c][:, ci]]).astype(float)
            y = segs[b][:, ci].astype(float)
            for m in methods:
                if m == "linear":
                    # burst마다 평균이 제거돼 있어 양 끝값이 이어지지 않는다 → 평균 0으로 가정한 선형 = 0과 같다. 끝점 기준 선형(비교용)만 둔다.
                    pred = np.interp(tb, [ta[-1], tc[0]], [segs[a][-1, ci], segs[c][0, ci]])
                elif m == "zero":
                    pred = np.zeros(len(tb))
                else:
                    pred = fill("sine", t_vis, y_vis, tb, seg_vis)
                pred = pred - pred.mean()
                rows.append({"seg_uid": b, "채널": ch, "방법": m, "span_s": span, "gap1_s": tb[0] - ta[-1] - 0.1,
                             "n": len(y), "sse": float(((y - pred) ** 2).sum()), "sst": float(((y - y.mean()) ** 2).sum())})
    return pd.DataFrame(rows)


def holdout_summary(h: pd.DataFrame, bins=(0, 8, 12, 16, 100)) -> pd.DataFrame:
    h = h.assign(구간=pd.cut(h["span_s"], bins=list(bins), right=False))
    g = h.groupby(["채널", "방법", "구간"], observed=True).agg(burst=("seg_uid", "nunique"), sse=("sse", "sum"), sst=("sst", "sum"))
    g["R2"] = 1 - g["sse"] / g["sst"]
    return g.reset_index()


# ------------------------------------------------------------------ 그림
def plot_gap_hist(gaps: pd.Series, ks, fig_dir, name="gap_hist.png"):
    import matplotlib.pyplot as plt
    import benchmark_jiw as B
    B._style()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(gaps, bins=np.arange(0, 17.5, 0.5), color="#8a8984", alpha=0.85, label="실제 burst 사이 공백")
    for k in ks:
        ax.axvline(k / FS, color="#d6332a", ls=":", lw=1)
    ax.axvline(float(gaps.median()), color="#2a78d6", lw=1.5, label=f"중앙값 {gaps.median():.1f}초")
    ax.set_xlabel("공백 길이(초). 빨간 점선 = 검증 A에서 가린 길이(0.5~3초)")
    ax.set_ylabel("공백 수")
    ax.set_title("실제 공백은 대부분 검증 A의 범위(≤ 3초)를 넘는다", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_mask_r2(me: pd.DataFrame, fig_dir, name="mask_r2.png"):
    import matplotlib.pyplot as plt
    import benchmark_jiw as B
    B._style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    col = {"zero": "#8a8984", "linear": "#2a78d6", "cubic": "#eb6834", "sine": "#d6332a"}
    for ax, ch in zip(axes, CH):
        for m in METHODS:
            d = me[(me["채널"] == ch) & (me["방법"] == m)]
            ax.plot(d["초"], d["R2"], marker="o", ms=4, color=col[m], label=METHOD_KO[m])
        ax.axhline(0.9, color=B.INK, ls="--", lw=1)
        ax.set_title(ch.replace("_", " "), fontsize=9.5)
        ax.set_xlabel("가린 길이(초)")
    axes[0].set_ylabel("가린 구간의 R² (0.9 = 합격선)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[0].set_ylim(-1.2, 1.05)
    fig.suptitle("검증 A: burst 안 마스킹 — 복원 R²", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_holdout_example(segs: dict, S: pd.DataFrame, fig_dir, name="holdout_example.png", n_ex: int = 3):
    """실제 공백 홀드아웃 예시: 전류에서 R²가 높은/중간/낮은 burst의 실제 vs 사인 복원"""
    import matplotlib.pyplot as plt
    import benchmark_jiw as B
    B._style()
    uids, st = list(S["seg_uid"]), S.set_index("seg_uid")["start_s"].to_dict()
    ci = 2
    cand = []
    for j in range(1, len(uids) - 1):
        a, b, c = uids[j - 1], uids[j], uids[j + 1]
        if min(len(segs[a]), len(segs[b]), len(segs[c])) < 30:
            continue
        ta, tb, tc = (st[u] + np.arange(len(segs[u])) / FS for u in (a, b, c))
        tv = np.concatenate([ta, tc])
        sv = np.r_[np.zeros(len(ta), dtype=int), np.ones(len(tc), dtype=int)]
        yv = np.concatenate([segs[a][:, ci], segs[c][:, ci]]).astype(float)
        p = sine_fit_predict(tv, yv, sv, tb)
        p = p - p.mean()
        y = segs[b][:, ci].astype(float)
        cand.append((1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), a, b, c, ta, tb, tc, p, y))
    cand.sort(key=lambda r: r[0])
    pick = [cand[-1], cand[len(cand) // 2], cand[0]][:n_ex]
    fig, axes = plt.subplots(1, len(pick), figsize=(11, 3.2))
    for ax, (r2, a, b, c, ta, tb, tc, p, y) in zip(axes, pick):
        ax.plot(ta, segs[a][:, ci], color="#8a8984", lw=1)
        ax.plot(tc, segs[c][:, ci], color="#8a8984", lw=1, label="보이는 burst")
        ax.plot(tb, y, color="#2a78d6", lw=1.6, label="가린 burst(실제)")
        ax.plot(tb, p, color="#d6332a", lw=1.6, ls="--", label="사인 복원")
        ax.set_title(f"{b}  R² {r2:.2f}  (span {tc[0] - ta[-1]:.1f}초)", fontsize=9)
        ax.set_xlabel("시간(초)")
    axes[0].set_ylabel("전류(평균 제거)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("검증 B 예시: 실제 공백을 건너 burst 하나를 복원 (좌 최고 · 중 중앙 · 우 최저 R²)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ 검증 C: burst 사이 위상 연속성 (전류)
def _fit_sine(t, y, f=None):
    """A·sin + B·cos + C 최소제곱. f가 없으면 F_GRID에서 탐색. 반환 (SSE, f, 계수)"""
    best = None
    for ff in (F_GRID if f is None else [f]):
        M = np.column_stack([np.sin(2 * np.pi * ff * t), np.cos(2 * np.pi * ff * t), np.ones(len(t))])
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        r = y - M @ c
        s = float(r @ r)
        if best is None or s < best[0]:
            best = (s, ff, c)
    return best


def burst_frequencies(segs: dict, S: pd.DataFrame, min_len: int = 40, ci: int = 2) -> pd.DataFrame:
    """burst마다 전류 사인을 맞춘 주파수·진폭·R² (burst 안의 규칙성)"""
    rows = []
    for u, st in zip(S["seg_uid"], S["start_s"]):
        x = segs[u][:, ci].astype(float)
        if len(x) < min_len:
            continue
        s, f, c = _fit_sine(st + np.arange(len(x)) / FS, x)
        rows.append({"seg_uid": u, "f_hz": f, "amp": float(np.hypot(c[0], c[1])), "r2": 1 - s / float(((x - x.mean()) ** 2).sum())})
    return pd.DataFrame(rows)


def phase_continuity(segs: dict, S: pd.DataFrame, min_len: int = 40, ci: int = 2) -> pd.DataFrame:
    """연속한 두 burst에서, 앞 burst의 사인(주파수·위상)을 공백 너머로 이어 다음 burst를 예측.
    phase_err = 다음 burst를 같은 주파수로 맞춘 위상 − 앞 burst 위상이 이어졌을 때의 위상 (라디안, −π~π). 위상이 이어지면 0 근처에 몰린다.
    r2_forward = 이어 붙인 사인으로 다음 burst(평균 제거)를 예측한 R²."""
    rows = []
    uids, st = list(S["seg_uid"]), S.set_index("seg_uid")["start_s"].to_dict()
    for a, b in zip(uids[:-1], uids[1:]):
        xa, xb = segs[a][:, ci].astype(float), segs[b][:, ci].astype(float)
        if len(xa) < min_len or len(xb) < min_len:
            continue
        ta, tb = st[a] + np.arange(len(xa)) / FS, st[b] + np.arange(len(xb)) / FS
        _, fa, ca = _fit_sine(ta, xa)
        pred = ca[0] * np.sin(2 * np.pi * fa * tb) + ca[1] * np.cos(2 * np.pi * fa * tb)
        pred = pred - pred.mean()
        M = np.column_stack([np.sin(2 * np.pi * fa * tb), np.cos(2 * np.pi * fa * tb), np.ones(len(tb))])
        cb, *_ = np.linalg.lstsq(M, xb, rcond=None)
        # 앞 burst 위상이 시간 tb[0]까지 이어졌다면 기대되는 (sin, cos) 계수 = ca (같은 절대 시간축을 썼으므로 그대로)
        e = (np.arctan2(cb[1], cb[0]) - np.arctan2(ca[1], ca[0]) + np.pi) % (2 * np.pi) - np.pi
        rows.append({"a": a, "b": b, "gap_s": tb[0] - ta[-1] - 1 / FS, "phase_err": float(e),
                     "r2_forward": 1 - float(((xb - pred) ** 2).sum()) / float(((xb - xb.mean()) ** 2).sum())})
    return pd.DataFrame(rows)


def plot_phase(pc: pd.DataFrame, fig_dir, name="phase_continuity.png"):
    import matplotlib.pyplot as plt
    import benchmark_jiw as B
    B._style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.hist(pc["phase_err"], bins=np.linspace(-np.pi, np.pi, 25), color="#8a8984", alpha=0.85)
    a1.axhline(len(pc) / 24, color="#d6332a", ls="--", lw=1, label="균등 분포(위상이 무작위일 때)")
    a1.set_xlabel("위상 오차(라디안). 0 근처에 몰리면 위상이 이어진 것")
    a1.set_ylabel("burst 쌍 수")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("burst 사이에서 전류 위상은 이어지지 않는다", loc="left", fontsize=10)
    a2.scatter(pc["gap_s"], np.abs(pc["phase_err"]), s=10, color="#2a78d6", alpha=0.6)
    a2.axhline(np.pi / 2, color="#d6332a", ls="--", lw=1, label="무작위 위상의 중앙값(π/2)")
    a2.set_xlabel("burst 사이 공백(초)")
    a2.set_ylabel("|위상 오차|(라디안)")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("공백이 짧아도(≤ 2.5초) 같다", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
