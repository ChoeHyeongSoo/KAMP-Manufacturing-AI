"""주제 ③ — JIW 33 조기경보(위험 상승 추세 감지) 합성 열화 시나리오 평가.

실제 이상은 1건이고 정상 → 이상 전이 구간이 없어 "N분 전 경보"를 실데이터로 검증할 수 없다(docs/final_design_JIW.md §2.4).
대신 test 정상 burst 열(time_block의 시간순 연속 구간)에 **강도가 서서히 커지는 합성 열화**를 넣고 경보가 열화 완성 몇 분 전에 울리는지 잰다.

- 경보 단계: 주의 = 위험 상승 추세 규칙(EWMA), 경보 = 28번의 CNN AND MCD (창 하나라도 두 모델이 동시에 임계 초과)
- 비교 기준(주의): 28번의 CNN 단독 초과(창 하나라도 임계 초과)
- 임계는 train 정상(time_block의 block < k)에서만 정한다(`evaluate.threshold_from_normal`과 같은 q99).
- 합성 열화는 24~28번의 `benchmark_jiw.inject`(4유형)를 burst마다 강도를 바꿔 쓴다. 열화 곡선과 유형은 우리가 정한 것이라
  결과는 "방법의 작동 확인"이지 실제 고장 예측 성능이 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import benchmark_jiw as B
import ensemble_jiw as E
import evaluate as ev
import paths

S_CRIT = 4.0          # 열화가 "실제 이상 수준"에 이르는 합성 강도 (28번 보정: 실제 이상은 강도 4~8 이상)
T0_MIN = 3.0          # 열화 시작 시각 (block 시작 후 분). 앞 3분은 정상 구간(거짓 주의율 측정 + EWMA 워밍업)
KINDS = B.INJ_TYPES
DURATIONS_MIN = (2.0, 5.0, 10.0)   # 열화 시작부터 강도 S_CRIT까지 걸리는 시간(분)
LAMBDAS = (0.1, 0.2, 0.4)
LAM_MAIN = 0.4          # 사후 선택: 열화 없는 정상 열의 거짓 사건 수가 CNN 단독과 같아지는 값 (표 §5에 세 값 모두 보고)
Q_TREND = 0.99


def seg_times() -> pd.Series:
    """세그먼트 시작 시각(초). 정상 파일의 첫 burst를 0으로 둔다."""
    S = pd.read_csv(paths.DATA_PROCESSED / "segments.csv", encoding="utf-8-sig")
    t = pd.to_datetime(S["start"])
    sec = (t - t[S["label"] == 0].min()).dt.total_seconds()
    return pd.Series(sec.to_numpy(), index=S["seg_uid"].to_numpy())


def burst_table(wins: pd.DataFrame, zc: np.ndarray, zm: np.ndarray, times: pd.Series) -> pd.DataFrame:
    """창별 z를 burst(세그먼트)별로 요약한다. z_and = 창마다 min(z_cnn, z_mcd)의 burst 안 최댓값(> 1이면 경보)."""
    d = pd.DataFrame({"seg_uid": wins["seg_uid"].to_numpy(), "zc": zc, "zm": zm})
    d["zand"] = np.minimum(d["zc"], d["zm"])
    g = d.groupby("seg_uid", sort=False)
    t = g.agg(zc_mean=("zc", "mean"), zc_max=("zc", "max"), zm_max=("zm", "max"), zand=("zand", "max"))
    t["t_sec"] = times.loc[t.index].to_numpy()
    return t.sort_values("t_sec").reset_index()


def ewma(u: np.ndarray, lam: float, init: float) -> np.ndarray:
    out = np.empty(len(u))
    s = init
    for i, v in enumerate(u):
        s = lam * v + (1 - lam) * s
        out[i] = s
    return out


def episodes(alert: np.ndarray, rearm: int = 3) -> np.ndarray:
    """연속 경보를 한 사건으로 센 시작 위치(bool). 경보가 rearm개 burst 연속 꺼져야 새 사건으로 센다."""
    start = np.zeros(len(alert), dtype=bool)
    quiet = rearm
    for i, a in enumerate(alert):
        if a:
            if quiet >= rearm:
                start[i] = True
            quiet = 0
        else:
            quiet += 1
    return start


def trend_threshold(train_burst: pd.DataFrame, lam: float, q: float = Q_TREND):
    """train 정상 burst 열의 log 평균 z에 EWMA를 씌워 q분위수를 임계로 쓴다. (임계, 초기값) 반환"""
    u = np.log(np.maximum(train_burst["zc_mean"].to_numpy(), 1e-6))
    init = float(np.median(u))
    return float(np.quantile(ewma(u, lam, init), q)), init


def collect_scenarios(segs, W, seeds=(0, 1, 2), win_s=2.0, kinds=KINDS, durations=DURATIONS_MIN, progress=True):
    """(burst 표, 임계 표). time_block 분할 k = 1..4 × 시드 × (정상 열 + 유형 × 열화 기간).

    burst 표 컬럼: k, seed, kind, dur_min, seg_uid, t_sec, sev, zc_mean, zc_max, zm_max, zand, t0, t_crit, t_end
    임계 표: k, seed, lam, thr, init
    """
    times = seg_times()
    Wv = W[W["win_s"] == win_s].reset_index(drop=True)
    rows, thr_rows = [], []
    splits = [(sp, k, tr, te) for sp, k, tr, te in B.iter_splits(Wv) if sp == "time_block"]
    bar = tqdm(total=len(splits) * len(seeds), desc="scenarios", dynamic_ncols=True, disable=not progress)
    for sp, k, tr, te in splits:
        n_te = te[te["label"] == 0]
        sd = np.concatenate([segs[u] for u in tr["seg_uid"].unique()]).std(0)
        seg_t = times.loc[n_te["seg_uid"].unique()].sort_values()
        t_start = float(seg_t.iloc[0])
        t0, t_end = t_start + T0_MIN * 60, float(seg_t.iloc[-1])
        # 주입 신호와 MCD 피처는 시드와 무관: 시나리오마다 한 번만 만든다
        sc = {("none", 0.0): (segs, B.recompute_features(segs, n_te, B.AMP_COLS), {u: 0.0 for u in seg_t.index})}
        for ki, kind in enumerate(kinds):
            for d in durations:
                sev = {u: float(np.clip(S_CRIT * (t - t0) / (d * 60), 0, S_CRIT)) for u, t in seg_t.items()}
                rng = np.random.default_rng([100 + k, ki, int(d)])
                inj = dict(segs)
                for u, s in sev.items():
                    if s > 0:
                        inj[u] = B.inject(segs[u], kind, s, sd, rng)
                sc[(kind, d)] = (inj, B.recompute_features(inj, n_te, B.AMP_COLS), sev)
        for seed in seeds:
            bar.set_postfix_str(f"k{k} s{seed}")
            cnn, mcd = E._load("cnn", win_s, sp, k, seed), E._load("mcd", win_s, sp, k, seed)
            s_tr_c, s_tr_m = cnn.score(segs, tr), mcd.score_features(tr)
            tc, tm = ev.threshold_from_normal(s_tr_c), ev.threshold_from_normal(s_tr_m)
            tb = burst_table(tr, s_tr_c / tc, s_tr_m / tm, times)
            for lam in LAMBDAS:
                thr, init = trend_threshold(tb, lam)
                thr_rows.append({"k": k, "seed": seed, "lam": lam, "thr": thr, "init": init})
            for (kind, d), (inj, F, sev) in sc.items():
                zc, zm = cnn.score(inj, n_te) / tc, mcd.score_features(F) / tm
                b = burst_table(n_te, zc, zm, times)
                b["sev"] = b["seg_uid"].map(sev)
                b = b.assign(k=k, seed=seed, kind=kind, dur_min=d, t0=t0, t_crit=t0 + d * 60 if kind != "none" else np.nan, t_end=t_end)
                rows.append(b)
            bar.update(1)
    bar.close()
    return pd.concat(rows, ignore_index=True), pd.DataFrame(thr_rows)


# ------------------------------------------------------------------ 규칙·평가
def alerts(b: pd.DataFrame, thr: pd.DataFrame, rule: str, lam: float = LAM_MAIN) -> np.ndarray:
    """한 시나리오(시간순 burst 표)의 주의/경보 bool 배열.
    rule: 'cnn'(CNN 단독 초과) · 'trend'(EWMA 추세) · 'alarm'(CNN AND MCD)"""
    if rule == "cnn":
        return (b["zc_max"] > 1).to_numpy()
    if rule == "alarm":
        return (b["zand"] > 1).to_numpy()
    r = thr[(thr["k"] == b["k"].iloc[0]) & (thr["seed"] == b["seed"].iloc[0]) & (thr["lam"] == lam)].iloc[0]
    u = np.log(np.maximum(b["zc_mean"].to_numpy(), 1e-6))
    return ewma(u, lam, r["init"]) > r["thr"]


def false_alert_rate(burst: pd.DataFrame, thr: pd.DataFrame, rule: str, lam: float = LAM_MAIN) -> pd.DataFrame:
    """열화가 없는 정상 열(kind == none)에서 시간당 거짓 주의·경보 사건 수"""
    rows = []
    for (k, seed), b in burst[burst["kind"] == "none"].groupby(["k", "seed"]):
        b = b.sort_values("t_sec")
        a = alerts(b, thr, rule, lam)
        hours = (b["t_sec"].max() - b["t_sec"].min()) / 3600
        rows.append({"k": k, "seed": seed, "events": int(episodes(a).sum()), "hours": hours,
                     "per_hour": episodes(a).sum() / hours, "burst_frac": float(a.mean())})
    return pd.DataFrame(rows)


def lead_times(burst: pd.DataFrame, thr: pd.DataFrame, rule: str, lam: float = LAM_MAIN) -> pd.DataFrame:
    """열화 시나리오마다 열화 시작(T0) 이후 첫 경보 시각과 열화 완성(T_crit) 대비 선행 시간(분, 양수 = 완성 전)"""
    rows = []
    for key, b in burst[burst["kind"] != "none"].groupby(["k", "seed", "kind", "dur_min"]):
        b = b.sort_values("t_sec").reset_index(drop=True)
        a = alerts(b, thr, rule, lam) & (b["t_sec"].to_numpy() >= b["t0"].iloc[0])
        hit = np.flatnonzero(a)
        t_first = float(b["t_sec"].iloc[hit[0]]) if len(hit) else np.nan
        rows.append(dict(zip(["k", "seed", "kind", "dur_min"], key),
                         t_first=t_first, lead_min=(b["t_crit"].iloc[0] - t_first) / 60 if len(hit) else np.nan,
                         since_t0_min=(t_first - b["t0"].iloc[0]) / 60 if len(hit) else np.nan,
                         sev_at_first=float(b["sev"].iloc[hit[0]]) if len(hit) else np.nan, detected=bool(len(hit))))
    return pd.DataFrame(rows)


def lead_summary(lead: pd.DataFrame) -> pd.DataFrame:
    g = lead.groupby("dur_min")
    out = pd.DataFrame({
        "시나리오 수": g.size(),
        "탐지율": g["detected"].mean(),
        "완성 전 경보율": g.apply(lambda d: float((d["lead_min"] > 0).mean())),
        "선행 시간 중앙값(분)": g["lead_min"].median(),
        "첫 경보 시점 합성 강도 중앙값": g["sev_at_first"].median(),
    })
    return out


def eta_errors(burst: pd.DataFrame, thr: pd.DataFrame, m: int = 10, lam: float = LAM_MAIN) -> pd.DataFrame:
    """주의(trend)가 처음 울린 시점에 최근 m개 burst의 log z_and를 직선으로 맞춰 경보(z_and = 1) 도달 시각을 외삽하고,
    실제 첫 경보 시각과 비교한다. 추정이 불가능한 경우(기울기 ≤ 0)는 eta 비어 있음."""
    rows = []
    for key, b in burst[burst["kind"] != "none"].groupby(["k", "seed", "kind", "dur_min"]):
        b = b.sort_values("t_sec").reset_index(drop=True)
        ok = b["t_sec"].to_numpy() >= b["t0"].iloc[0]
        att = np.flatnonzero(alerts(b, thr, "trend", lam) & ok)
        alm = np.flatnonzero(alerts(b, thr, "alarm") & ok)
        if not len(att) or not len(alm):
            continue
        i = att[0]
        lo = max(0, i - m + 1)
        t = b["t_sec"].to_numpy()[lo:i + 1]
        y = np.log(np.maximum(b["zand"].to_numpy()[lo:i + 1], 1e-6))
        if len(t) < 3 or np.ptp(t) == 0:
            continue
        slope, icpt = np.polyfit(t - t[-1], y, 1)
        t_alarm = float(b["t_sec"].iloc[alm[0]])
        eta_s = (0 - icpt) / slope if slope > 0 else np.nan
        rows.append(dict(zip(["k", "seed", "kind", "dur_min"], key), t_attention=float(t[-1]), t_alarm=t_alarm,
                         true_remaining_min=(t_alarm - t[-1]) / 60, eta_min=eta_s / 60 if slope > 0 else np.nan))
    d = pd.DataFrame(rows)
    d["abs_err_min"] = (d["eta_min"] - d["true_remaining_min"]).abs()
    return d


# ------------------------------------------------------------------ 그림 · 대시보드
def plot_timeline(burst: pd.DataFrame, thr: pd.DataFrame, fig_dir, k: int = 2, seed: int = 0, kind: str = "current",
                  dur: float = 5.0, name: str = "timeline.png"):
    """열화 시나리오 하나의 시간 흐름: burst별 CNN z, EWMA 추세와 임계, 첫 주의·경보, 열화 시작·완성 시각"""
    import matplotlib.pyplot as plt
    B._style()
    plt.rcParams["axes.formatter.use_mathtext"] = True   # 로그 축 눈금의 위첨자 마이너스가 한글 글꼴에 없어 깨지는 것을 막는다
    b = burst[(burst["k"] == k) & (burst["seed"] == seed) & (burst["kind"] == kind) & (burst["dur_min"] == dur)].sort_values("t_sec")
    t = (b["t_sec"].to_numpy() - b["t0"].iloc[0]) / 60
    r = thr[(thr["k"] == k) & (thr["seed"] == seed) & (thr["lam"] == LAM_MAIN)].iloc[0]
    u = np.log(np.maximum(b["zc_mean"].to_numpy(), 1e-6))
    ew = ewma(u, LAM_MAIN, r["init"])
    a_cnn, a_tr, a_al = alerts(b, thr, "cnn"), alerts(b, thr, "trend"), alerts(b, thr, "alarm")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax1.semilogy(t, np.maximum(b["zc_max"], 1e-3), ".-", color="#2a78d6", lw=0.8, ms=4, label="CNN z (burst 최댓값)")
    ax1.semilogy(t, np.maximum(b["zm_max"], 1e-3), ".-", color="#8a8984", lw=0.8, ms=4, label="MCD z (burst 최댓값)")
    ax1.axhline(1, color=B.INK, ls="--", lw=1)
    ax1.plot(t[a_al], b["zand"].to_numpy()[a_al], "v", color="#d6332a", ms=7, label="경보 (CNN AND MCD)")
    ax1.set_ylabel("점수 / 임계")
    ax2.plot(t, ew, color="#eb6834", lw=1.4, label="위험 상승 추세 (EWMA of log z)")
    ax2.axhline(r["thr"], color=B.INK, ls="--", lw=1, label="추세 임계 (train 정상 q99)")
    ax2.plot(t[a_tr], ew[a_tr], "^", color="#eb6834", ms=6, label="추세 주의")
    ax2.set_ylabel("추세 통계")
    ax2.set_xlabel("열화 시작 후 시간(분)")
    t_crit = (b["t_crit"].iloc[0] - b["t0"].iloc[0]) / 60
    for ax in (ax1, ax2):
        ax.axvline(0, color="#52514e", lw=1)
        ax.axvline(t_crit, color="#d6332a", lw=1, ls=":")
    first = lambda a: t[a & (t >= 0)][0] if (a & (t >= 0)).any() else np.nan
    ax1.set_title(f"합성 열화 시나리오 ({B.INJ_KO[kind]}, {dur:g}분 동안 강도 0 → {S_CRIT:g}) — 첫 CNN 주의 {first(a_cnn):.1f}분, "
                  f"추세 주의 {first(a_tr):.1f}분, 경보 {first(a_al):.1f}분, 열화 완성 {t_crit:.1f}분", loc="left", fontsize=9)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_lead(leads: dict, fig_dir, name: str = "lead_time.png"):
    """규칙별·열화 기간별 선행 시간(분) 분포. leads = {규칙 이름: lead_times 표}"""
    import matplotlib.pyplot as plt
    B._style()
    durs = sorted(next(iter(leads.values()))["dur_min"].unique())
    fig, axes = plt.subplots(1, len(durs), figsize=(10, 3.6))
    colors = ["#2a78d6", "#eb6834", "#d6332a", "#8a8984"]
    for ax, d in zip(axes, durs):
        data = [v[v["dur_min"] == d]["lead_min"].dropna().to_numpy() for v in leads.values()]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "black"})
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(leads) + 1), list(leads), rotation=20, ha="right", fontsize=8)
        ax.axhline(0, color=B.INK, lw=1)
        ax.axhline(d, color="#d6332a", ls=":", lw=1)
        ax.set_title(f"열화 {d:g}분", fontsize=9)
    axes[0].set_ylabel("열화 완성 전 선행 시간(분)")
    fig.suptitle("경보 규칙별 선행 시간 (점선 = 열화 시작 시점, 0 = 열화 완성)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def write_dashboard(fig_dir, out_path, tables: dict, cards: list[str], note: str):
    """정적 HTML 대시보드(서버 불필요). fig_dir의 PNG를 base64로 넣고 표·근거 카드 문장을 붙인다."""
    import base64
    import html
    img = lambda n: ("<img style='max-width:100%' src='data:image/png;base64," +
                     base64.b64encode((fig_dir / n).read_bytes()).decode() + "'/>") if (fig_dir / n).exists() else ""
    parts = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>프레스 유압펌프 조기경보 대시보드</title>"
             "<style>body{font-family:'Malgun Gothic',sans-serif;max-width:980px;margin:24px auto;color:#222;padding:0 16px}"
             "h2{border-bottom:1px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;font-size:13px}"
             "td,th{border:1px solid #ccc;padding:3px 8px}th{background:#f4f3ef}.card{background:#f7f6f2;border-left:4px solid #d6332a;"
             "padding:8px 12px;margin:8px 0;font-size:13px}.note{color:#666;font-size:12px}</style></head><body>",
             "<h1>프레스 유압펌프 이상 조기경보 대시보드</h1>", f"<p class='note'>{html.escape(note)}</p>",
             "<h2>1. 합성 열화 시나리오 한 건의 시간 흐름</h2>", img("timeline.png"),
             "<h2>2. 경보 규칙별 선행 시간</h2>", img("lead_time.png")]
    for title, df in tables.items():
        parts += [f"<h2>{html.escape(title)}</h2>", df.to_html(float_format=lambda v: f"{v:.2f}")]
    parts.append("<h2>근거 카드 예시 (실제 이상 창·오경보 창)</h2>")
    parts += [f"<div class='card'>{html.escape(t)}</div>" for t in cards]
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
