"""주제 ③ — JIW 28 앙상블(시계열 모델 + 통계 모델 결합) 점수 수집·결합 규칙·평가.

저장된 가중치(24·25·26 노트북)만 불러 쓰고 새로 학습하지 않는다.

- 관점 A(원 신호 시퀀스): CNN(DeepAnT) / LSTM-AD / MTAD-GAT 예측 오차. 관점 B(윈도우 통계): MCD [amp 18열].
- 결합 규칙은 점수를 각 모델 임계(train 정상 q99)로 나눈 z = s / thr 위에서 정의한다. 경보는 z > 1.
    단독 z_a, AND = min(z_a, z_b), OR = max(z_a, z_b).
  임계는 결합 후 다시 구하지 않고 각 모델의 값을 그대로 쓴다("AND-고정", docs/design_JIW.md §2 사실 B).
- 임계 모드: `global` = 정상 전체 q99, `state` = 운전 상태별 q99. 상태는 윈도우 전류 RMS로 추정한다(STATE_CUR_RMS_BOUND).
  사후 라벨(`state` 열)은 평가용 층화에만 쓰고 추론 입력으로는 쓰지 않는다.
- 합성 이상은 test 정상 세그먼트에 주입하고, 임계는 train 정상 q99를 그대로 쓴다(운영 임계 기준 탐지율).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import benchmark_jiw as B
import evaluate as ev
import models_jiw as M
import paths

NB_SRC = {"cnn": ("24_model_forecasting_JIW", "cnn_deepant", M.CNNDeepAnT),
          "lstm": ("24_model_forecasting_JIW", "lstm_ad", M.LSTMAD),
          "gat": ("25_model_graph_JIW", "mtad_gat", M.MTADGAT)}
MCD_SRC = ("26_model_distribution_JIW", "mcd_amp")
ALL_MODELS = ["cnn", "lstm", "gat", "mcd"]
STATE_CUR_RMS_BOUND = 125.2          # 21 노트북의 2-means 경계(정상 고부하 중앙값 157, 저부하 중앙값 86 사이)
BURSTS_PER_HOUR = 3600.0 / ev.BURST_GAP_S
KEYS = ["split", "fold", "seed"]


def est_state(wins: pd.DataFrame) -> np.ndarray:
    """윈도우 전류 RMS로 추정한 운전 상태 (0 = 고부하, 1 = 저부하)"""
    return np.where(wins["AI2_Current_rms"].to_numpy() > STATE_CUR_RMS_BOUND, 0, 1)


def _load(model: str, win_s: float, sp: str, k: int, seed: int):
    if model == "mcd":
        nb, mid = MCD_SRC
        return M.MCD.load(paths.MODELS / nb / f"{mid}_w{win_s:g}s_{sp}_f{k}_s0.joblib")   # MCD 가중치는 시드 0만 저장돼 있다
    nb, mid, cls = NB_SRC[model]
    return cls.load(paths.MODELS / nb / f"{mid}_w{win_s:g}s_{sp}_f{k}_s{seed}.pt")


def _score(model, m, segs, wins, feats=None):
    if model == "mcd":
        return m.score_features(wins if feats is None else feats)
    return m.score(segs, wins)


def collect(segs, W, models=ALL_MODELS, seeds=(0, 1, 2), win_s=2.0, severities=B.SEVERITIES, progress=True):
    """(real, inj) 표를 반환한다.

    real: test 윈도우(정상 + 이상)의 모델별 점수 s_<m>, 모델별 임계 thr_g_<m>(전체)·thr_s_<m>(추정 상태별), 추정 상태.
    inj : test 정상 윈도우에 합성 이상(유형 × 강도)을 넣은 같은 구조의 표.
    """
    Wv = W[W["win_s"] == win_s].reset_index(drop=True)
    real_rows, inj_rows = [], []
    splits = list(B.iter_splits(Wv))
    bar = tqdm(total=len(splits) * len(seeds), desc="collect", dynamic_ncols=True, disable=not progress)
    for sp, k, tr, te in splits:
        y = te["label"].to_numpy()
        n_te = te[y == 0]
        st_tr, st_te, st_n = est_state(tr), est_state(te), est_state(n_te)
        sd = np.concatenate([segs[u] for u in tr["seg_uid"].unique()]).std(0)
        feats = {}
        for kind in B.INJ_TYPES:   # 주입 신호와 그 피처는 시드와 무관하므로 분할마다 한 번만 만든다
            for sev in severities:
                rng = np.random.default_rng(100 + k)
                inj = dict(segs)
                for u in n_te["seg_uid"].unique():
                    inj[u] = B.inject(segs[u], kind, sev, sd, rng)
                feats[(kind, sev)] = (inj, B.recompute_features(inj, n_te, B.AMP_COLS))
        for seed in seeds:
            bar.set_postfix_str(f"{sp} f{k} s{seed}")
            loaded = {m: _load(m, win_s, sp, k, seed) for m in models}
            base = {"split": sp, "fold": k, "seed": seed}
            real = te[["seg_uid", "t_start", "label", "state", "vib_grade", "cur_grade"]].reset_index(drop=True).assign(
                est_state=st_te, **base)
            injs = {kv: n_te[["seg_uid", "t_start"]].reset_index(drop=True).assign(
                est_state=st_n, kind=kv[0], sev=kv[1], **base) for kv in feats}
            for m in models:
                s_tr = _score(m, loaded[m], segs, tr)
                tg = ev.threshold_from_normal(s_tr)
                ts = {s: ev.threshold_from_normal(s_tr[st_tr == s]) if (st_tr == s).sum() >= 30 else tg for s in (0, 1)}
                real[f"s_{m}"] = _score(m, loaded[m], segs, te)
                real[f"thr_g_{m}"] = tg
                real[f"thr_s_{m}"] = np.where(st_te == 0, ts[0], ts[1])
                for kv, (inj, F) in feats.items():
                    d = injs[kv]
                    d[f"s_{m}"] = _score(m, loaded[m], inj, n_te, F)
                    d[f"thr_g_{m}"] = tg
                    d[f"thr_s_{m}"] = np.where(st_n == 0, ts[0], ts[1])
            real_rows.append(real)
            inj_rows.extend(injs.values())
            bar.update(1)
    bar.close()
    return pd.concat(real_rows, ignore_index=True), pd.concat(inj_rows, ignore_index=True)


# ------------------------------------------------------------------ 결합 규칙
def z_scores(df: pd.DataFrame, model: str, mode: str = "global") -> np.ndarray:
    t = df[f"thr_{'g' if mode == 'global' else 's'}_{model}"].to_numpy()
    return df[f"s_{model}"].to_numpy() / t


def rule_z(df: pd.DataFrame, rule: str, mode: str = "global") -> np.ndarray:
    """rule: 'cnn' | 'mcd' | 'cnn&mcd' | 'cnn|mcd' 처럼 모델 이름과 &(AND) · |(OR). 경보는 z > 1."""
    if "&" in rule:
        return np.minimum.reduce([z_scores(df, m, mode) for m in rule.split("&")])
    if "|" in rule:
        return np.maximum.reduce([z_scores(df, m, mode) for m in rule.split("|")])
    return z_scores(df, rule, mode)


def seg_alarm(df: pd.DataFrame, alarm: np.ndarray) -> pd.Series:
    return pd.Series(alarm, index=df.index).groupby([df["seg_uid"], df["seed"], df["split"], df["fold"]]).any()


# ------------------------------------------------------------------ 평가
def evaluate_rule(real: pd.DataFrame, inj: pd.DataFrame, rule: str, mode: str = "global", win_s: float = 2.0,
                  kofn=(1, 1)) -> pd.DataFrame:
    """(split, fold, seed)별 지표 표. kofn=(k, n)이면 세그먼트 안 시간순으로 최근 n개 중 k개 초과일 때만 경보."""
    k_, n_ = kofn
    rows = []
    z_all = rule_z(real, rule, mode)
    zi_all = rule_z(inj, rule, mode)
    real = real.assign(_z=z_all, _hit=z_all > 1)
    inj = inj.assign(_hit=zi_all > 1)
    if (k_, n_) != (1, 1):
        real = real.sort_values(KEYS + ["seg_uid", "t_start"])
        real["_hit"] = real.groupby(KEYS + ["seg_uid"])["_hit"].transform(lambda s: ev.kofn_alarm(s, k_, n_))
        inj = inj.sort_values(KEYS + ["kind", "sev", "seg_uid", "t_start"])
        inj["_hit"] = inj.groupby(KEYS + ["kind", "sev", "seg_uid"])["_hit"].transform(lambda s: ev.kofn_alarm(s, k_, n_))
    for key, g in real.groupby(KEYS, sort=False):
        y = g["label"].to_numpy()
        n, o = g[y == 0], g[y == 1]
        dly = ev.detection_delay_s(o["_hit"].astype(float), o["seg_uid"], o["t_start"], 0.5, win_s=win_s)
        det = dly["detected"].to_numpy(bool)
        gi = inj[(inj["split"] == key[0]) & (inj["fold"] == key[1]) & (inj["seed"] == key[2])]
        row = dict(zip(KEYS, key))
        row.update({
            "auc": ev.auc(y, g["_z"].to_numpy()),
            "fpr_sample": float(n["_hit"].mean()),
            "fpr_segment": float(n.groupby("seg_uid")["_hit"].any().mean()),
            "fpr_high_load": float(n.loc[n["state"] == 0, "_hit"].mean()) if (n["state"] == 0).any() else np.nan,
            "fpr_low_load": float(n.loc[n["state"] == 1, "_hit"].mean()) if (n["state"] == 1).any() else np.nan,
            "delay_median_s": float(np.nanmedian(dly["delay_in_window_s"])) if det.any() else np.nan,
            "n_out_seg": len(dly), "n_detected": int(det.sum()), "recall_window": float(o["_hit"].mean()),
            "inj_mean": float(gi["_hit"].mean()),
        })
        for kind, gk in gi.groupby("kind"):
            row[f"inj_{kind}"] = float(gk["_hit"].mean())
        for sev, gs in gi.groupby("sev"):
            row[f"inj_sev{sev:g}"] = float(gs["_hit"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rules(real, inj, rules: dict, win_s: float = 2.0, progress=True) -> pd.DataFrame:
    """rules: {이름: (rule, mode, (k, n))} → 이름 열을 붙인 cv 표"""
    out = []
    for name, (rule, mode, kofn) in tqdm(rules.items(), desc="rules", disable=not progress):
        out.append(evaluate_rule(real, inj, rule, mode, win_s, kofn).assign(model=name, rule=rule, mode=mode,
                                                                        k=kofn[0], n=kofn[1], win_s=win_s))
    return pd.concat(out, ignore_index=True)


def to_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    """results/README 공통 컬럼: (model, split, fold)별 시드 평균 + fold='mean' 행"""
    cols = ["auc", "fpr_sample", "fpr_segment", "delay_median_s", "n_out_seg", "n_detected", "win_s"]
    per = cv.groupby(["model", "split", "fold"], sort=False)[cols].mean().reset_index()
    mean = per.groupby(["model", "split"], sort=False)[cols].mean().reset_index().assign(fold="mean")
    out = pd.concat([per, mean], ignore_index=True)
    return out[ev.METRIC_COLS + ["n_out_seg", "n_detected", "win_s"]]


def summary(cv: pd.DataFrame) -> pd.DataFrame:
    sev_cols = sorted((c for c in cv.columns if c.startswith("inj_sev")), key=lambda c: float(c[7:]))
    cols = ["auc", "fpr_sample", "fpr_segment", "fpr_high_load", "fpr_low_load", "recall_window", "n_detected", "n_out_seg",
            "delay_median_s", "inj_mean"] + sev_cols
    out = cv.groupby(["model", "split"], sort=False)[cols].mean()
    out["fp_per_hour"] = out["fpr_segment"] * BURSTS_PER_HOUR
    return out


# ------------------------------------------------------------------ H1: 세그먼트 단위 쌍 부트스트랩
def seg_fp_table(real: pd.DataFrame, rule: str, mode: str = "global") -> pd.DataFrame:
    """정상 test 세그먼트별 오경보 여부(시드 평균). 분할 안에서 모든 정상 세그먼트가 한 번씩만 test에 들어간다."""
    n = real[real["label"] == 0]
    hit = pd.Series(rule_z(n, rule, mode) > 1, index=n.index)
    t = hit.groupby([n["split"], n["seed"], n["seg_uid"]]).any().astype(float)
    return t.groupby(level=["split", "seg_uid"]).mean().rename(rule)


def paired_bootstrap(real: pd.DataFrame, base: str, other: str, split: str, mode: str = "global",
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """세그먼트를 복원추출해 (other − base) 세그먼트 오경보율 차이의 95% 구간을 구한다."""
    a = seg_fp_table(real, base, mode).xs(split, level="split")
    b = seg_fp_table(real, other, mode).xs(split, level="split")
    d = (b - a).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"split": split, "base": base, "other": other, "mode": mode, "n_seg": len(d),
            "fp_base": float(a.mean()), "fp_other": float(b.mean()), "diff": float(d.mean()),
            "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975))}


# ------------------------------------------------------------------ 상태 추정 정확도 (H4 보조)
def state_estimate_accuracy(real: pd.DataFrame) -> pd.DataFrame:
    n = real[(real["label"] == 0) & real["state"].notna()].drop_duplicates(["split", "fold", "seg_uid", "t_start"])
    n = n.assign(ok=(n["est_state"] == n["state"]))
    return n.groupby("state")["ok"].agg(["mean", "size"]).rename(index={0.0: "고부하(0)", 1.0: "저부하(1)"})


# ------------------------------------------------------------------ 그림
def plot_quadrant(real: pd.DataFrame, fig_dir, a: str = "cnn", b: str = "mcd", split: str = "group_kfold_seg",
                  seed: int = 0, name: str = "quadrant.png"):
    """두 모델의 z 점수(점수/임계) 산점도. 오른쪽 위 사분면이 AND 경보, 오른쪽 + 위쪽 전체가 OR 경보."""
    import matplotlib.pyplot as plt
    B._style()
    plt.rcParams["axes.formatter.use_mathtext"] = True   # 로그 축 눈금의 위첨자 마이너스가 한글 글꼴에 없어 깨지는 것을 막는다
    g = real[(real["split"] == split) & (real["seed"] == seed)]
    za, zb = z_scores(g, a), z_scores(g, b)
    y = g["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for lab, col, nm in [(0, "#8a8984", "정상"), (1, "#d6332a", "실제 이상")]:
        ax.scatter(za[y == lab], zb[y == lab], s=9 if lab == 0 else 16, alpha=0.45 if lab == 0 else 0.8, color=col, label=nm,
                   edgecolors="none")
    ax.axvline(1, color=B.INK, ls="--", lw=1)
    ax.axhline(1, color=B.INK, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{a.upper()} 점수 / 임계 (1 = 임계)")
    ax.set_ylabel(f"{b.upper()} 점수 / 임계 (1 = 임계)")
    ax.set_title("두 관점의 점수: 정상은 한쪽만 넘고, 실제 이상은 둘 다 크게 넘는다", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_tradeoff(cv: pd.DataFrame, fig_dir, name: str = "tradeoff.png", sev: str = "inj_sev1"):
    """x = 세그먼트 오경보율, y = 합성 이상(강도 1) 탐지율. 왼쪽 위가 좋다."""
    import matplotlib.pyplot as plt
    B._style()
    d = cv.groupby("model", sort=False)[["fpr_segment", sev]].mean()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for m, r in d.iterrows():
        col = "#2a78d6" if ("&" not in m and "|" not in m and "mcd" not in m) else "#eb6834" if "&" in m else "#8a8984"
        ax.scatter(r["fpr_segment"], r[sev], s=48, color=col, edgecolors="white", zorder=3)
        ax.annotate(m, (r["fpr_segment"], r[sev]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("세그먼트 오경보율 (정상 세그먼트 중 한 번이라도 경보)")
    ax.set_ylabel("합성 이상(강도 1) 탐지율")
    ax.set_title("오경보와 약한 이상 탐지는 맞바꿈 관계 (두 분할·시드 평균)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_strength(cv: pd.DataFrame, fig_dir, models: list, name: str = "strength.png"):
    """합성 이상 강도별 탐지율. AND가 단독 모델을 따라잡는 강도를 본다."""
    import matplotlib.pyplot as plt
    B._style()
    sev = sorted((c for c in cv.columns if c.startswith("inj_sev")), key=lambda c: float(c[7:]))
    x = [float(c[7:]) for c in sev]
    d = cv[cv["model"].isin(models)].groupby("model")[sev].mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in models:
        ax.plot(x, d.loc[m], marker="o", ms=4, lw=1.6, label=m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.set_xlabel("합성 이상 강도 (클수록 뚜렷)")
    ax.set_ylabel("창 단위 탐지율")
    ax.set_title("약한 이상에서는 AND가 크게 떨어지고, 강한 이상에서 따라잡는다", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_state(cv: pd.DataFrame, fig_dir, models: list, name: str = "state_fpr.png"):
    """운전 상태별 오경보율: 임계 모드(전체 vs 상태별)를 나란히"""
    import matplotlib.pyplot as plt
    B._style()
    d = cv[cv["model"].isin(models)].groupby("model")[["fpr_high_load", "fpr_low_load"]].mean().loc[models]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(d) + 2), 4))
    ax.bar(x - 0.2, d["fpr_high_load"], 0.38, color="#eb6834", label="고부하 (state 0)", edgecolor="white")
    ax.bar(x + 0.2, d["fpr_low_load"], 0.38, color="#2a78d6", label="저부하 (state 1)", edgecolor="white")
    ax.axhline(0.01, color=B.INK, ls="--", lw=1)
    ax.set_xticks(x, d.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("정상 윈도우 오경보율")
    ax.set_title("운전 상태별 오경보율", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
