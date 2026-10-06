"""22번 고전 시계열 모델 공통 실행·평가.

기존 24~26 벤치마크와 같은 데이터·분할·임계·합성 이상 정의를 사용하되,
PyTorch가 없는 CPU 환경에서도 네 고전 모델만 독립 실행할 수 있게 분리한다.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_quality as dq
import evaluate as ev
import features
from models_jsc import FEATURE_MODELS, FS, SEQ_MODELS
import paths
import preprocess

STATE_NAME = {0: "고부하", 1: "저부하"}
INJ_TYPES = ("spike", "amplitude", "antiphase", "current")
INJ_KO = {"spike": "진동 스파이크", "amplitude": "진폭 증가", "antiphase": "반대 흔들림", "current": "전류 불규칙"}
SEVERITIES = (0.25, 0.5, 1.0, 2.0)
AMP_COLS = [f"{ch}_{key}" for ch in preprocess.SENSORS for key in features.AMP_KEYS]


def load_inputs() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """표준 mean 전처리 세그먼트 배열과 11번 윈도우 계약을 읽는다."""
    df = pd.concat(dq.load_all().values(), ignore_index=True)
    pre = preprocess.preprocess(df)
    segs = {uid: g[preprocess.SENSORS].to_numpy(np.float32) for uid, g in pre.groupby("seg_uid", sort=False)}
    wins = pd.read_parquet(paths.DATA_PROCESSED / "11_window_features.parquet")
    return segs, wins


def iter_splits(wins: pd.DataFrame):
    """기존 계약: GroupKFold 5 + 정상 forward time-block 4(이상 전체 반복 평가)."""
    for fold in range(5):
        train = wins[(wins["fold"] != fold) & (wins["label"] == 0)]
        test = wins[wins["fold"] == fold]
        yield "group_kfold_seg", fold, train, test
    normal = wins[wins["label"] == 0]
    for fold in range(1, 5):
        train = normal[normal["block"] < fold]
        test = pd.concat([normal[normal["block"] == fold], wins[wins["label"] == 1]])
        yield "time_block", fold, train, test


def inject(x: np.ndarray, kind: str, severity: float, sd: np.ndarray, rng) -> np.ndarray:
    """24~26과 같은 네 합성 이상. 주입 후 평균을 다시 제거한다."""
    x = np.asarray(x, dtype=float).copy()
    if kind == "spike":
        mask = rng.random(len(x)) < 0.10
        x[mask, 0] += rng.choice([-1, 1], mask.sum()) * severity * 3 * sd[0]
    elif kind == "amplitude":
        x[:, :2] *= 1 + 0.5 * severity
    elif kind == "antiphase":
        x[:, 1] -= 0.5 * severity * x[:, 0] * sd[1] / sd[0]
    elif kind == "current":
        x[:, 2] += rng.normal(0, 0.2 * severity * sd[2], len(x))
    else:
        raise ValueError(kind)
    return (x - x.mean(axis=0)).astype(np.float32)


def amp_features(segs: dict[str, np.ndarray], wins: pd.DataFrame) -> pd.DataFrame:
    """주입 세그먼트에서 원본 parquet과 동일한 진폭 18열을 다시 계산한다."""
    win = int(round(float(wins["win_s"].iloc[0]) * FS))
    starts = np.round(wins["t_start"].to_numpy(dtype=float) * FS).astype(int)
    x = np.stack([segs[uid][start:start + win] for uid, start in zip(wins["seg_uid"], starts)]).astype(float)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    peak = np.max(np.abs(x), axis=1)
    p2p = np.max(x, axis=1) - np.min(x, axis=1)
    centered = x - x.mean(axis=1, keepdims=True)
    m2 = np.mean(centered ** 2, axis=1)
    safe = np.where(m2 > 0, m2, np.nan)
    kurt = np.mean(centered ** 4, axis=1) / safe ** 2 - 3.0
    skew = np.mean(centered ** 3, axis=1) / safe ** 1.5
    crest = np.divide(peak, rms, out=np.full_like(peak, np.nan), where=rms > 0)
    values = {"rms": rms, "peak": peak, "p2p": p2p, "kurt": kurt, "crest": crest, "skew": skew}
    out = {}
    for idx, ch in enumerate(preprocess.SENSORS):
        for key in features.AMP_KEYS:
            out[f"{ch}_{key}"] = values[key][:, idx]
    return pd.DataFrame(out)[AMP_COLS]


class _Runner:
    def __init__(self, key: str, win_s: float, lookback: int):
        self.key = key
        if key in SEQ_MODELS:
            self.cls, self.seq = SEQ_MODELS[key], True
            self.args = (lookback, int(round(win_s * FS)))
        else:
            self.cls, self.seq = FEATURE_MODELS[key], False
            self.args = (AMP_COLS,)

    def fit_or_load(self, train, segs, seed, path, retrain):
        if path.exists() and not retrain:
            self.model = self.cls.load(path)
            return "loaded"
        self.model = self.cls(*self.args)
        if self.seq:
            uids = train["seg_uid"].drop_duplicates().tolist()
            self.model.fit([segs[uid] for uid in uids], seed=seed)
        else:
            self.model.fit_features(train, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return "trained"

    def score(self, wins, segs):
        return self.model.score(segs, wins) if self.seq else self.model.score_features(wins)


def run(keys, nb: str, win_list=(1.0,), seed=0, lookback=5, retrain=False, segs=None, wins=None):
    """네 모델 × 윈도우 × 9분할을 실행해 상세 점수·오류 사례를 반환한다."""
    if segs is None or wins is None:
        segs, wins = load_inputs()
    rows, errors, score_rows = [], [], []
    attr = wins.groupby("seg_uid")[["state", "vib_grade", "cur_grade"]].first()
    for win_s in win_list:
        wv = wins[wins["win_s"] == win_s].reset_index(drop=True)
        for split_name, fold, train, test in iter_splits(wv):
            for key in keys:
                runner = _Runner(key, win_s, lookback)
                model_id = f"{key}_amp_w{win_s:g}s" if not runner.seq else f"{key}_w{win_s:g}s"
                path = paths.MODELS / nb / f"{model_id}_{split_name}_f{fold}_s{seed}.{runner.cls.ext}"
                started = time.perf_counter()
                status = runner.fit_or_load(train, segs, seed, path, retrain)
                train_score = runner.score(train, segs)
                test_score = runner.score(test, segs)
                threshold = ev.threshold_from_normal(train_score)

                y = test["label"].to_numpy(dtype=int)
                normal, outlier = test[y == 0], test[y == 1]
                score_normal, score_outlier = test_score[y == 0], test_score[y == 1]
                delay = ev.detection_delay_s(
                    score_outlier, outlier["seg_uid"], outlier["t_start"], threshold, win_s=win_s
                )
                detected = delay["detected"].to_numpy(dtype=bool)
                state = normal["state"].to_numpy()
                row = {
                    "family": runner.cls.family,
                    "model": runner.cls.name if runner.seq else f"{runner.cls.name} [amp]",
                    "key": key,
                    "model_id": model_id,
                    "win_s": win_s,
                    "split": split_name,
                    "fold": fold,
                    "seed": seed,
                    "status": status,
                    "auc": ev.auc(y, test_score),
                    "fpr_sample": ev.fpr_sample(score_normal, threshold),
                    "fpr_segment": ev.fpr_segment(score_normal, normal["seg_uid"], threshold),
                    "delay_median_s": float(np.nanmedian(delay["delay_in_window_s"])) if detected.any() else np.nan,
                    "felt_delay_median_s": float(np.nanmedian(delay["delay_in_window_s"])) + ev.BURST_GAP_S if detected.any() else np.nan,
                    "thr": threshold,
                    "n_out_seg": len(delay),
                    "n_detected": int(detected.sum()),
                    "recall_window": float(np.mean(score_outlier > threshold)),
                    "fpr_high_load": float(np.mean(score_normal[state == 0] > threshold)) if (state == 0).any() else np.nan,
                    "fpr_low_load": float(np.mean(score_normal[state == 1] > threshold)) if (state == 1).any() else np.nan,
                }

                # 실제 이상 라벨과 독립적인 약한 합성 이상 평가.
                injection_threshold = float(np.quantile(score_normal, 0.99))
                train_sd = np.concatenate([segs[uid] for uid in train["seg_uid"].unique()]).std(axis=0)
                test_uids = normal["seg_uid"].unique()
                for kind in INJ_TYPES:
                    for severity in SEVERITIES:
                        rng = np.random.default_rng(1000 + fold)
                        injected = dict(segs)
                        for uid in test_uids:
                            injected[uid] = inject(segs[uid], kind, severity, train_sd, rng)
                        if runner.seq:
                            injected_score = runner.model.score(injected, normal)
                        else:
                            injected_score = runner.model.score_features(amp_features(injected, normal))
                        row[f"inj_{kind}_{severity:g}"] = float(np.mean(injected_score > injection_threshold))
                row["inj_mean"] = float(np.mean([
                    row[f"inj_{kind}_{severity:g}"] for kind in INJ_TYPES for severity in SEVERITIES
                ]))
                row["sec"] = round(time.perf_counter() - started, 3)
                rows.append(row)

                exceed = pd.DataFrame({"seg_uid": normal["seg_uid"].to_numpy(), "exceed": score_normal > threshold})
                fp = exceed.groupby("seg_uid")["exceed"].agg(["sum", "size"]).query("sum > 0")
                for uid, values in fp.iterrows():
                    errors.append({
                        "model_id": model_id, "split": split_name, "fold": fold, "type": "FP", "seg_uid": uid,
                        "n_exceed": int(values["sum"]), "n_windows": int(values["size"]),
                        "state": STATE_NAME.get(attr.loc[uid, "state"], ""),
                    })
                for uid in delay.loc[~detected, "seg_uid"]:
                    errors.append({
                        "model_id": model_id, "split": split_name, "fold": fold, "type": "FN", "seg_uid": uid,
                        "vib_grade": attr.loc[uid, "vib_grade"], "cur_grade": attr.loc[uid, "cur_grade"],
                    })
                score_rows.append(pd.DataFrame({
                    "model_id": model_id, "split": split_name, "fold": fold,
                    "seg_uid": test["seg_uid"].to_numpy(), "label": y, "state": test["state"].to_numpy(),
                    "t_start": test["t_start"].to_numpy(), "score": test_score, "thr": threshold,
                }))
    return pd.DataFrame(rows), pd.DataFrame(errors), pd.concat(score_rows, ignore_index=True)


def to_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    """results/README의 공통 7열 + 모델 노트북 보조 열."""
    extra = ["win_s", "thr", "n_out_seg", "n_detected", "felt_delay_median_s"]
    values = ["auc", "fpr_sample", "fpr_segment", "delay_median_s"] + extra
    per = cv.groupby(["model_id", "split", "fold"], sort=False)[values].mean().reset_index()
    mean = per.groupby(["model_id", "split"], sort=False)[values].mean().reset_index().assign(fold="mean")
    out = pd.concat([per, mean], ignore_index=True).rename(columns={"model_id": "model"})
    return out[ev.METRIC_COLS + extra]


def summary(cv: pd.DataFrame) -> pd.DataFrame:
    values = ["auc", "fpr_sample", "fpr_segment", "n_detected", "n_out_seg", "delay_median_s",
              "fpr_high_load", "fpr_low_load", "inj_mean", "sec"]
    return cv.groupby(["model", "split"], sort=False)[values].mean().round(4)


def _style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e5e4e0", "axes.axisbelow": True,
    })


def plot_overview(cv: pd.DataFrame, fig_dir, name="metrics_overview.png"):
    _style()
    models = list(dict.fromkeys(cv["model"]))
    metrics = (("auc", "AUC"), ("fpr_segment", "세그먼트 오경보율"), ("n_detected", "이상 탐지 세그먼트"), ("inj_mean", "합성 이상 탐지율"))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    x = np.arange(len(models))
    for row, split_name in enumerate(("group_kfold_seg", "time_block")):
        data = cv[cv["split"] == split_name]
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            values = data.groupby("model")[metric].mean().reindex(models)
            errors = data.groupby("model")[metric].std().reindex(models).fillna(0)
            ax.bar(x, values, yerr=errors, color="#2a78d6", edgecolor="white", capsize=2)
            ax.set_xticks(x, models, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{split_name} · {title}", loc="left", fontsize=9)
            ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_state_fpr(cv: pd.DataFrame, fig_dir, name="fpr_by_state.png"):
    _style()
    data = cv.groupby("model", sort=False)[["fpr_high_load", "fpr_low_load"]].mean()
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, data["fpr_high_load"], 0.38, label="고부하", color="#eb6834")
    ax.bar(x + 0.2, data["fpr_low_load"], 0.38, label="저부하", color="#2a78d6")
    ax.axhline(0.01, color="#52514e", ls="--", lw=1)
    ax.set_xticks(x, data.index, rotation=20, ha="right")
    ax.set_ylabel("정상 윈도우 오경보율")
    ax.set_title("운전 상태별 오경보율", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_injection(cv: pd.DataFrame, fig_dir, name="injection.png"):
    _style()
    data = []
    for model, group in cv.groupby("model", sort=False):
        row = {"model": model}
        for kind in INJ_TYPES:
            row[kind] = group[[f"inj_{kind}_{severity:g}" for severity in SEVERITIES]].mean(axis=1).mean()
        data.append(row)
    table = pd.DataFrame(data).set_index("model")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.5, float(table.max().max())), aspect="auto")
    for i in range(len(table)):
        for j in range(len(INJ_TYPES)):
            ax.text(j, i, f"{table.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(len(INJ_TYPES)), [INJ_KO[k] for k in INJ_TYPES])
    ax.set_yticks(range(len(table)), table.index)
    ax.set_title("합성 이상 유형별 탐지율", loc="left")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_score_margin(scores: pd.DataFrame, fig_dir, name="score_margin.png"):
    _style()
    models = list(dict.fromkeys(scores["model_id"]))
    groups = (("정상 고부하", 0, 0, "#9a9993"), ("정상 저부하", 0, 1, "#eda100"), ("이상", 1, None, "#e34948"))
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(models) + 2))
    for i, model in enumerate(models):
        base = scores[scores["model_id"] == model]
        for j, (label, y, state, color) in enumerate(groups):
            part = base[base["label"] == y]
            if state is not None:
                part = part[part["state"] == state]
            ratio = np.clip(part["score"] / part["thr"], 1e-4, None)
            ax.boxplot(ratio, positions=[i + (j - 1) * 0.23], widths=0.19, vert=False, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": color, "edgecolor": "white"},
                       medianprops={"color": "black"})
    ax.axvline(1, color="#e34948", ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(models)), models)
    ax.set_ylim(len(models) - 0.5, -0.6)
    ax.set_xlabel("점수 / 임계값 — 1 초과 시 경보")
    ax.set_title("정상·이상 점수의 임계 여유", loc="left")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=g[3]) for g in groups]
    ax.legend(handles, [g[0] for g in groups], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
