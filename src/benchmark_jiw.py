"""주제 ③ — JIW 모델 노트북(24·25·26) 공통 실행·평가·시각화.

팀 계약(docs/modeling_kickoff.md)을 그대로 따른다.
- 입력: `preprocess.preprocess()` 결과 + `11_window_features.parquet`의 윈도우(`seg_uid`, `t_start`, `win_s`)
- 분할: `group_kfold_seg`(parquet `fold` 0~4), `time_block`(parquet `block`, train = block < k, test = block k + 이상 전체)
- 임계: train 정상 점수 q99 (`evaluate.threshold_from_normal`)
- 지표: `evaluate`의 auc · fpr_sample · fpr_segment · detection_delay_s

추가 평가(심사 3·5번)
- 운전 상태별 오경보율: `state` 0 = 고부하, 1 = 저부하 (02 §1)
- 합성 이상 주입: test 정상 세그먼트에 EDA 징후 4종을 강도 4단계로 넣고, test 정상 오경보율 1% 임계에서 탐지율을 잰다.
  실제 이상은 1건이라 상위 모델이 AUC 1.0에 몰리므로 약한(초기) 이상 탐지력으로 모델을 가른다.
  주입 후 세그먼트 평균을 다시 빼서 표준 전처리와 같은 조건을 유지한다.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import data_quality as dq
import evaluate as ev
import features
import paths
import preprocess
from models_jiw import FEATURE_MODELS, FS, SEQ_MODELS

STATE_NAME = {0: "고부하", 1: "저부하"}
INJ_TYPES = ["spike", "amplitude", "antiphase", "current"]
INJ_KO = {"spike": "진동 스파이크", "amplitude": "진폭 증가", "antiphase": "상·하부 반대 흔들림", "current": "전류 불규칙"}
SEVERITIES = [0.25, 0.5, 1.0, 2.0]
AMP_COLS = [f"{ch}_{k}" for ch in preprocess.SENSORS for k in features.AMP_KEYS]
SHAPE_COLS = [f"{ch}_shape_{k}" for ch in preprocess.SENSORS for k in features.SHAPE_KEYS]
SINE_COLS = list(features.SINE_KEYS)
REL_COLS = list(features.REL_KEYS)
# 착수 가이드 §7 피처군 ablation: 기본 = 진폭, 최소 보고 = 기본 · 기본+사인
FEATURE_SETS = {"amp": AMP_COLS, "amp_sine": AMP_COLS + SINE_COLS, "amp_shape": AMP_COLS + SHAPE_COLS,
                "amp_rel": AMP_COLS + REL_COLS}
BASELINE_CSV = paths.RESULTS / "21_model_rule_baseline_CHS" / "metrics.csv"
BASELINE_RULES = {"rule1_rms_ai0": "21 rule1", "rule3_rms_state_dir": "21 rule3"}   # 크기 규칙 · 최강 규칙


# ------------------------------------------------------------------ data
def load_inputs():
    """(세그먼트 배열 dict, 윈도우 표). 세그먼트는 표준 전처리 결과, 윈도우 표는 11 parquet 전체."""
    df = pd.concat(dq.load_all().values(), ignore_index=True)
    pre = preprocess.preprocess(df)
    segs = {u: g[preprocess.SENSORS].to_numpy(np.float32) for u, g in pre.groupby("seg_uid", sort=False)}
    W = pd.read_parquet(paths.DATA_PROCESSED / "11_window_features.parquet")
    return segs, W


def iter_splits(Wv: pd.DataFrame):
    """(split, fold, train 정상 윈도우, test 윈도우) — Wv는 한 윈도우 길이의 표"""
    for k in range(5):
        yield "group_kfold_seg", k, Wv[(Wv["fold"] != k) & (Wv["label"] == 0)], Wv[Wv["fold"] == k]
    nb = Wv[Wv["label"] == 0]
    for k in range(1, 5):
        yield "time_block", k, nb[nb["block"] < k], pd.concat([nb[nb["block"] == k], Wv[Wv["label"] == 1]])


# ------------------------------------------------------------------ synthetic anomaly injection
def inject(x: np.ndarray, kind: str, sev: float, sd: np.ndarray, rng) -> np.ndarray:
    """x: 전처리된 (n, 3) 세그먼트, sd: train 정상 채널 표준편차. 반환은 세그먼트 평균을 다시 뺀 값."""
    x = x.astype(float).copy()
    if kind == "spike":        # AI0 샘플 10%에 충격
        m = rng.random(len(x)) < 0.10
        x[m, 0] += rng.choice([-1, 1], m.sum()) * sev * 3 * sd[0]
    elif kind == "amplitude":  # 진동 진폭 증가
        x[:, :2] *= 1 + 0.5 * sev
    elif kind == "antiphase":  # 상·하부 반대 방향 성분 (시소 흔들림)
        x[:, 1] -= 0.5 * sev * x[:, 0] * sd[1] / sd[0]
    elif kind == "current":    # 전류 불규칙(백색잡음)
        x[:, 2] += rng.normal(0, 0.2 * sev * sd[2], len(x))
    else:
        raise ValueError(kind)
    return (x - x.mean(0)).astype(np.float32)


def recompute_features(segs: dict, wins: pd.DataFrame, cols=None) -> pd.DataFrame:
    """세그먼트 배열에서 parquet과 같은 정의의 피처를 다시 계산한다 (주입 신호용).
    진폭·형상은 윈도우마다, 사인 잔차·채널 관계는 세그먼트당 1회 계산해 브로드캐스트한다(features 모듈과 동일)."""
    cols = list(cols or AMP_COLS)
    need_seg = any(c in SINE_COLS + REL_COLS for c in cols)
    seg_feat = {}
    if need_seg:
        for u in wins["seg_uid"].unique():
            x = segs[u]
            d = features.current_sine_features(x[:, 2], FS) if any(c in SINE_COLS for c in cols) else {}
            if any(c in REL_COLS for c in cols):
                d.update(features.relation_features(pd.DataFrame(x, columns=preprocess.SENSORS)))
            seg_feat[u] = d
    w = int(round(wins["win_s"].iloc[0] * FS))
    rows = []
    for u, t in zip(wins["seg_uid"], wins["t_start"]):
        s = int(round(t * FS))
        x = segs[u][s:s + w]
        row = {}
        for i, ch in enumerate(preprocess.SENSORS):
            row.update({f"{ch}_{k}": v for k, v in features.amplitude_features(x[:, i]).items()})
            if any("_shape_" in c for c in cols):
                row.update({f"{ch}_shape_{k}": v for k, v in features.shape_features(x[:, i]).items()})
        row.update(seg_feat.get(u, {}))
        if "vib_rms_ratio" in cols:   # 윈도우 단위 관계 피처 (features.window_features와 같은 정의)
            r1 = row[f"{preprocess.SENSORS[1]}_rms"]
            row["vib_rms_ratio"] = row[f"{preprocess.SENSORS[0]}_rms"] / r1 if r1 > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)[cols]


def amp_features(segs: dict, wins: pd.DataFrame) -> pd.DataFrame:
    """진폭 피처 18열만 다시 계산 (recompute_features의 기본값)."""
    return recompute_features(segs, wins, AMP_COLS)


# ------------------------------------------------------------------ one model on one split
class _Runner:
    """시계열 모델과 피처 모델을 같은 방식으로 학습·채점하게 감싼다."""

    def __init__(self, key, win_s, lookback, cols):
        self.key, self.win_s = key, win_s
        if key in SEQ_MODELS:
            self.cls, self.seq = SEQ_MODELS[key], True
            self.args = (lookback, int(round(win_s * FS)))
        else:
            self.cls, self.seq = FEATURE_MODELS[key], False
            self.args = (cols,)

    def fit_or_load(self, tr, segs, seed, epochs, path, retrain, progress):
        if path.exists() and not retrain:
            self.m = self.cls.load(path)
            return "loaded"
        self.m = self.cls(*self.args)
        if self.seq:
            self.m.fit([segs[u] for u in tr["seg_uid"].unique()], seed=seed, epochs=epochs, progress=progress)
        else:
            self.m.fit_features(tr, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.m.save(path)
        return "trained"

    def score(self, wins, segs):
        if self.seq:
            return self.m.score(segs, wins)
        return self.m.score_features(wins)

    def score_injected(self, wins, segs_inj, feats=None):
        """feats: 주입 신호로 다시 계산한 피처 표(피처 모델용, 없으면 여기서 계산)"""
        if self.seq:
            return self.m.score(segs_inj, wins)
        return self.m.score_features(feats if feats is not None else recompute_features(segs_inj, wins, self.args[0]))


def _seg_attr(W):
    g = W.groupby("seg_uid")
    return g[["state", "vib_grade", "cur_grade"]].first()


def run(keys, nb, win_list=(1.0, 2.0), seeds=(0, 1, 2), epochs=30, lookback=5, feature_sets=None,
        retrain=False, progress=True, segs=None, W=None):
    """모델 × 윈도우 × 분할 × 시드 전체를 돌려 (cv_scores, error_cases, scores) 표를 반환한다.

    - 가중치: `models/<nb>/<model_id>_<split>_f<fold>_s<seed>.(pt|joblib)`. 있으면 불러오고(retrain=False) 없으면 학습 후 저장.
    - feature_sets: 피처 모델의 입력 열 조합 {이름: 열 목록} (기본 {"amp": 진폭 18열}). 시계열 모델에는 무관.
      model_id = 시계열 `<key>_w<초>s`, 피처 `<key>_<조합>_w<초>s`.
    - error_cases: 첫 시드 기준 FP(초과 윈도우가 있는 정상 세그먼트)·FN(미탐지 이상 세그먼트) 목록.
    - scores: 첫 시드 기준 윈도우별 점수(점수 분포·임계 여유 그림용).
    """
    if segs is None or W is None:
        segs, W = load_inputs()
    feature_sets = feature_sets or {"amp": AMP_COLS}
    attr = _seg_attr(W)
    fcache = {}   # 주입 신호 피처는 모델·조합과 무관하므로 (윈도우, 분할, fold, 유형, 강도)별로 전체 열을 한 번만 계산
    all_cols = list(dict.fromkeys(c for cols in feature_sets.values() for c in cols))
    jobs = []
    for win_s in win_list:
        Wv = W[W["win_s"] == win_s].reset_index(drop=True)
        for sp, k, tr, te in iter_splits(Wv):
            for key in keys:
                for fs in ([None] if key in SEQ_MODELS else list(feature_sets)):
                    for seed in (seeds if key in SEQ_MODELS else seeds[:1]):
                        jobs.append((win_s, sp, k, tr, te, key, fs, seed))
    rows, errs, scores = [], [], []
    bar = tqdm(jobs, desc=nb, dynamic_ncols=True, disable=not progress)
    for win_s, sp, k, tr, te, key, fs, seed in bar:
        model_id = f"{key}_w{win_s:g}s" if fs is None else f"{key}_{fs}_w{win_s:g}s"
        bar.set_postfix_str(f"{model_id} {sp} f{k} s{seed}")
        r = _Runner(key, win_s, lookback, feature_sets.get(fs))
        path = paths.MODELS / nb / f"{model_id}_{sp}_f{k}_s{seed}.{r.cls.ext}"
        t0 = time.time()
        status = r.fit_or_load(tr, segs, seed, epochs, path, retrain, progress)
        s_tr, s_te = r.score(tr, segs), r.score(te, segs)
        thr = ev.threshold_from_normal(s_tr)
        y = te["label"].to_numpy()
        n_te, o_te = te[y == 0], te[y == 1]
        sn, so = s_te[y == 0], s_te[y == 1]
        dly = ev.detection_delay_s(so, o_te["seg_uid"], o_te["t_start"], thr, win_s=win_s)
        det = dly["detected"].to_numpy(bool)
        st = n_te["state"].to_numpy()
        row = {"family": r.cls.family, "model": r.cls.name if fs is None else f"{r.cls.name} [{fs}]", "key": key,
               "feature_set": fs or "", "model_id": model_id, "win_s": win_s, "split": sp, "fold": k, "seed": seed, "status": status,
               "auc": ev.auc(y, s_te), "fpr_sample": ev.fpr_sample(sn, thr),
               "fpr_segment": ev.fpr_segment(sn, n_te["seg_uid"], thr),
               "delay_median_s": float(np.nanmedian(dly["delay_in_window_s"])) if det.any() else np.nan,
               "felt_delay_median_s": float(np.nanmedian(dly["delay_in_window_s"])) + ev.BURST_GAP_S if det.any() else np.nan,
               "thr": thr, "n_out_seg": len(dly), "n_detected": int(det.sum()),
               "recall_window": float(np.mean(so > thr)),
               "fpr_high_load": float(np.mean(sn[st == 0] > thr)) if (st == 0).any() else np.nan,
               "fpr_low_load": float(np.mean(sn[st == 1] > thr)) if (st == 1).any() else np.nan}
        # 합성 이상: test 정상 오경보율 1% 임계
        t1 = np.quantile(sn, 0.99)
        sd = np.concatenate([segs[u] for u in tr["seg_uid"].unique()]).std(0)
        te_uids = n_te["seg_uid"].unique()
        for kind in INJ_TYPES:
            for sev in SEVERITIES:
                ck = (win_s, sp, k, kind, sev)
                if r.seq or ck not in fcache:
                    rng = np.random.default_rng(100 + k)
                    inj = dict(segs)
                    for u in te_uids:
                        inj[u] = inject(segs[u], kind, sev, sd, rng)
                    if not r.seq:
                        fcache[ck] = recompute_features(inj, n_te, all_cols)
                s_inj = r.score_injected(n_te, inj) if r.seq else r.score_injected(n_te, None, fcache[ck][feature_sets[fs]])
                row[f"inj_{kind}_{sev:g}"] = float(np.mean(s_inj > t1))
        row["inj_mean"] = float(np.mean([row[f"inj_{kd}_{v:g}"] for kd in INJ_TYPES for v in SEVERITIES]))
        row["sec"] = round(time.time() - t0, 1)
        rows.append(row)
        if seed == seeds[0]:
            ex = pd.DataFrame({"seg_uid": n_te["seg_uid"].to_numpy(), "exceed": sn > thr})
            fp = ex.groupby("seg_uid")["exceed"].agg(["sum", "size"]).query("sum > 0")
            for u, v in fp.iterrows():
                errs.append({"model_id": row["model_id"], "split": sp, "fold": k, "type": "FP", "seg_uid": u,
                             "n_exceed": int(v["sum"]), "n_windows": int(v["size"]),
                             "state": STATE_NAME.get(attr.loc[u, "state"], "")})
            for u in dly.loc[~det, "seg_uid"]:
                errs.append({"model_id": row["model_id"], "split": sp, "fold": k, "type": "FN", "seg_uid": u,
                             "vib_grade": attr.loc[u, "vib_grade"], "cur_grade": attr.loc[u, "cur_grade"]})
            scores.append(pd.DataFrame({"model_id": row["model_id"], "split": sp, "fold": k,
                                        "seg_uid": te["seg_uid"].to_numpy(), "label": y,
                                        "state": te["state"].to_numpy(), "score": s_te, "thr": thr}))
    return pd.DataFrame(rows), pd.DataFrame(errs), pd.concat(scores, ignore_index=True)


def to_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    """results/README 공통 컬럼 표: (model_id, split, fold)별 시드 평균 + fold='mean' 행.
    공통 7열 뒤에 21 metrics.csv와 같은 보조 열(win_s, thr, n_out_seg, n_detected, felt_delay_median_s)을 붙인다."""
    extra = ["win_s", "thr", "n_out_seg", "n_detected", "felt_delay_median_s"]
    cols = ["auc", "fpr_sample", "fpr_segment", "delay_median_s"] + extra
    per = cv.groupby(["model_id", "split", "fold"], sort=False)[cols].mean().reset_index()
    mean = per.groupby(["model_id", "split"], sort=False)[cols].mean().reset_index().assign(fold="mean")
    out = pd.concat([per, mean], ignore_index=True).rename(columns={"model_id": "model"})
    return out[ev.METRIC_COLS + extra]


def summary(cv: pd.DataFrame) -> pd.DataFrame:
    cols = ["auc", "fpr_sample", "fpr_segment", "recall_window", "n_detected", "n_out_seg", "delay_median_s",
            "fpr_high_load", "fpr_low_load", "inj_mean"]
    return cv.groupby(["model", "win_s", "split"], sort=False)[cols].mean().round(3)


# ------------------------------------------------------------------ figures
WIN_COLOR = {1.0: "#2a78d6", 2.0: "#eb6834"}
INK = "#52514e"


def _style():
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e4e0",
                         "axes.axisbelow": True, "font.size": 9})


def baseline_table() -> pd.DataFrame:
    """21 규칙 베이스라인의 fold 평균 행 (비교표·그림 기준선용)"""
    if not BASELINE_CSV.exists():
        return pd.DataFrame()
    b = pd.read_csv(BASELINE_CSV)
    return b[b["fold"].astype(str) == "mean"].reset_index(drop=True)


def plot_overview(cv, fig_dir, name="metrics_overview.png"):
    """분할(행) × 지표(열) 막대: 모델별 1초·2초 윈도우, fold·시드 평균 ± 표준편차, 점선 = 21 rule3(윈도우별 색).
    모델이 4개보다 많으면(26) 가로 막대로 그려 모델 이름을 세로축에 둔다."""
    _style()
    models = list(dict.fromkeys(cv["model"]))
    base = baseline_table()
    mets = [("auc", "AUC (높을수록 좋음)"), ("fpr_sample", "fpr_sample (낮을수록 좋음)"),
            ("fpr_segment", "fpr_segment (낮을수록 좋음)"), ("inj_mean", "합성 이상 탐지율 (높을수록 좋음)")]
    many = len(models) > 4
    if many:
        fig, axes = plt.subplots(2, 4, figsize=(18, 2 * (0.32 * len(models) + 1.2)), sharey=True)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(17, 6.8))
    x = np.arange(len(models))
    for i, sp in enumerate(["group_kfold_seg", "time_block"]):
        d = cv[cv["split"] == sp]
        for j, (m, title) in enumerate(mets):
            ax = axes[i, j]
            for o, w in zip([-0.2, 0.2], [1.0, 2.0]):
                g = d[d["win_s"] == w].groupby("model")[m]
                mu, sd = g.mean().reindex(models), g.std().reindex(models)
                kw = dict(color=WIN_COLOR[w], label=f"{w:g}초 윈도우", edgecolor="white",
                          error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 2})
                if many:
                    ax.barh(x + o, mu, 0.38, xerr=sd, **kw)
                else:
                    ax.bar(x + o, mu, 0.38, yerr=sd, **kw)
            if len(base) and m in base.columns:   # 21 최강 규칙(rule3) 윈도우별 기준선
                for w in (1.0, 2.0):
                    b = base[(base["split"] == sp) & (base["model"] == "rule3_rms_state_dir") & (base["win_s"] == w)]
                    if len(b):
                        (ax.axvline if many else ax.axhline)(b[m].iloc[0], color=WIN_COLOR[w], ls="--", lw=1, alpha=0.8)
            if many:
                ax.set_yticks(x, models, fontsize=8)
                ax.set_ylim(len(models) - 0.5, -0.5)
                ax.grid(axis="y", visible=False)
            else:
                ax.set_xticks(x, models)
                ax.grid(axis="x", visible=False)
            ax.set_title(f"{sp} · {title}", loc="left", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.99, 0.005, "점선 = 21 rule3 (윈도우별 색)", fontsize=8, color=INK, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_state_fpr(cv, fig_dir, name="fpr_by_state.png"):
    """정상 윈도우의 운전 상태별 오경보율 (두 분할·시드 평균)"""
    _style()
    d = cv.groupby(["model", "win_s"], sort=False)[["fpr_high_load", "fpr_low_load"]].mean()
    labels = [f"{m}\n{w:g}초" for m, w in d.index]
    x = np.arange(len(d))
    many = len(d) > 8
    if many:
        labels = [l.replace("\n", " ") for l in labels]
    fig, ax = plt.subplots(figsize=(min(1.3 * len(d) + 2, 26), 5.5 if many else 3.8))
    ax.bar(x - 0.2, d["fpr_high_load"], 0.38, color="#eb6834", label="고부하 (state 0)", edgecolor="white")
    ax.bar(x + 0.2, d["fpr_low_load"], 0.38, color="#2a78d6", label="저부하 (state 1)", edgecolor="white")
    ax.axhline(0.01, color=INK, ls="--", lw=1)
    if many:
        ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    else:
        ax.set_xticks(x, labels)
    ax.set_ylabel("정상 윈도우 오경보율")
    ax.set_title("운전 상태별 오경보율 (train 정상 q99 임계, 두 분할 평균)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_injection(cv, fig_dir, name="injection.png"):
    """왼쪽: 모델·윈도우 × 이상 유형 탐지율 히트맵, 오른쪽: 강도별 평균 탐지율"""
    _style()
    for kd in INJ_TYPES:
        cv[f"inj_{kd}"] = cv[[f"inj_{kd}_{v:g}" for v in SEVERITIES]].mean(axis=1)
    rows = cv.groupby(["model", "win_s"], sort=False)
    M = rows[[f"inj_{kd}" for kd in INJ_TYPES] + ["inj_mean"]].mean()
    labels = [f"{m} {w:g}초" for m, w in M.index]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(M) + 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=max(0.5, M.values.max()), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 * im.get_clim()[1] else "#0b0b0b")
    ax.set_xticks(range(M.shape[1]), [INJ_KO[k] for k in INJ_TYPES] + ["평균"])
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    ax.set_title("합성 이상 탐지율 (test 정상 오경보율 1% 임계, 강도 평균)", loc="left", fontsize=10)
    ax = axes[1]
    styles = ["-", "--", ":", "-."]
    top = set(M["inj_mean"].nlargest(6).index) if len(M) > 8 else set(M.index)   # 선이 많으면 평균 상위 6개만
    for i, (lab, (m, w)) in enumerate(zip(labels, M.index)):
        if (m, w) not in top:
            continue
        d = cv[(cv["model"] == m) & (cv["win_s"] == w)]
        yv = [d[[f"inj_{kd}_{v:g}" for kd in INJ_TYPES]].mean(axis=1).mean() for v in SEVERITIES]
        ax.plot(SEVERITIES, yv, marker="o", ms=5, lw=2, ls=styles[i // 2 % 4], color=WIN_COLOR[w], label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEVERITIES, [f"{v:g}" for v in SEVERITIES])
    ax.set_ylim(0, 1)
    ax.set_xlabel("주입 강도")
    ax.set_ylabel("탐지율")
    ax.set_title("강도별 탐지율 (4유형 평균" + (", 평균 상위 6개)" if len(M) > 8 else ")"), loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_score_margin(scores, fig_dir, name="score_margin.png"):
    """점수 ÷ 임계값 분포(로그): 정상 고부하 / 정상 저부하 / 이상. 1보다 크면 경보."""
    _style()
    ids = list(dict.fromkeys(scores["model_id"]))
    groups = [("정상 · 고부하", (scores["label"] == 0) & (scores["state"] == 0), "#9a9993"),
              ("정상 · 저부하", (scores["label"] == 0) & (scores["state"] == 1), "#eda100"),
              ("이상", scores["label"] == 1, "#e34948")]
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(ids) + 1.5))
    for i, mid in enumerate(ids):
        d = scores["model_id"] == mid
        for j, (lab, mask, c) in enumerate(groups):
            s = scores[d & mask]
            shift = min(0.0, float(scores.loc[d, "score"].min()))
            v = np.clip((s["score"] - shift) / (s["thr"] - shift), 1e-3, None)
            ax.boxplot(v, positions=[i + (j - 1) * 0.25], widths=0.2, vert=False, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": c, "edgecolor": "white"},
                       medianprops={"color": "#0b0b0b"})
    ax.axvline(1, color="#e34948", ls="--", lw=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(range(len(ids)), ids)
    ax.set_ylim(len(ids) - 0.5, -0.7)
    ax.set_xlabel("점수 ÷ 임계값 (로그) — 1보다 크면 경보")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in groups], [g[0] for g in groups],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_title("점수 분포와 임계 여유 (첫 시드, 전체 fold)", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
