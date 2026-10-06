"""23번 Ridge[first_difference] + MCD[diff_amp] 실시간 앙상블 검증.

두 모델 모두 34번에서 확정한 causal 1차 차분 신호를 사용한다.

- Ridge-VAR: 과거 5개 차분 샘플로 다음 차분 샘플을 예측한다.
- MCD: 같은 차분 신호의 1초 진폭 피처 18개에 강건 공분산을 적합한다.
- 각 점수를 자기 train 정상 q99로 나눈 뒤 단독·AND·OR·평균 결합을 비교한다.

임계·MCD·Ridge alpha는 fold의 학습 정상만 사용하며 이상 라벨은 선택에 사용하지 않는다.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

import benchmark_jsc as bj
import evaluate as ev
from models_jsc import FS, RidgeVAR
import paths
import preprocess
import realtime_jsc as rt

SEEDS = (0, 1, 2)
RULES = ("ridge_diff", "mcd_diff_amp", "ridge&mcd", "ridge|mcd", "ridge+mcd[cal]")
RULE_NAME = {
    "ridge_diff": "Ridge diff",
    "mcd_diff_amp": "MCD diff_amp",
    "ridge&mcd": "Ridge AND MCD",
    "ridge|mcd": "Ridge OR MCD",
    "ridge+mcd[cal]": "Ridge+MCD calibrated mean",
}
STATE_NAME = {0: "고부하", 1: "저부하"}


class RobustMCD:
    """train 중앙값 대치·표준화 뒤 Minimum Covariance Determinant 거리."""

    def __init__(self, cols: list[str], seed: int = 0):
        self.cols = list(cols)
        self.seed = int(seed)

    def fit(self, frame: pd.DataFrame):
        self.fill = frame[self.cols].astype(float).median()
        x = frame[self.cols].astype(float).fillna(self.fill).to_numpy()
        self.scaler = StandardScaler().fit(x)
        self.detector = MinCovDet(random_state=self.seed).fit(self.scaler.transform(x))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.cols].astype(float).fillna(self.fill).to_numpy()
        return self.detector.mahalanobis(self.scaler.transform(x))

    def details(self) -> dict:
        return {
            "seed": self.seed,
            "n_features": len(self.cols),
            "support_size": int(np.asarray(self.detector.support_).sum()),
            "support_fraction": float(np.asarray(self.detector.support_).mean()),
        }


def first_difference_arrays(segs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """세그먼트별 1차 차분. 첫 샘플은 0이고 경계를 절대 넘지 않는다."""
    out = {}
    for uid, value in segs.items():
        x = np.asarray(value, dtype=np.float32)
        d = np.zeros_like(x)
        if len(x) > 1:
            d[1:] = x[1:] - x[:-1]
        out[uid] = d
    return out


def load_inputs():
    """형식 통일 신호·offline mean·first difference와 1초 윈도우를 읽는다."""
    _, unified, wins = rt.load_inputs()
    unified_segs = rt.segment_arrays(unified)
    mean_segs, _ = rt.prepare_variant(unified, "offline_mean")
    diff_segs, _ = rt.prepare_variant(unified, "first_difference")
    return unified_segs, mean_segs, diff_segs, wins


def rule_scores(z_ridge: np.ndarray, z_mcd: np.ndarray, mean_thr: float) -> dict[str, np.ndarray]:
    """개별 q99=1 기준으로 결합한 점수. 모든 규칙의 경보 경계는 1이다."""
    z_ridge = np.asarray(z_ridge, dtype=float)
    z_mcd = np.asarray(z_mcd, dtype=float)
    return {
        "ridge_diff": z_ridge,
        "mcd_diff_amp": z_mcd,
        "ridge&mcd": np.minimum(z_ridge, z_mcd),
        "ridge|mcd": np.maximum(z_ridge, z_mcd),
        "ridge+mcd[cal]": (0.5 * (z_ridge + z_mcd)) / mean_thr,
    }


def _state_fpr(normal: pd.DataFrame, hit: np.ndarray) -> dict[str, float]:
    table = normal[["seg_uid", "state"]].copy()
    table["hit"] = np.asarray(hit, dtype=bool)
    out = {}
    for state, name in ((0, "high_load"), (1, "low_load")):
        part = table[table["state"] == state]
        out[f"fpr_sample_{name}"] = float(part["hit"].mean()) if len(part) else np.nan
        per_seg = part.groupby("seg_uid", sort=False)["hit"].any()
        out[f"fpr_segment_{name}"] = float(per_seg.mean()) if len(per_seg) else np.nan
    return out


def _injected_inputs(
    mean_segs: dict[str, np.ndarray],
    normal: pd.DataFrame,
    train_uids: list[str],
    fold: int,
) -> dict[tuple[str, float], tuple[dict[str, np.ndarray], pd.DataFrame]]:
    """시드와 무관한 합성 차분 신호·진폭 피처를 분할마다 한 번 만든다."""
    train_sd = np.concatenate([mean_segs[uid] for uid in train_uids]).std(axis=0)
    test_uids = normal["seg_uid"].drop_duplicates().tolist()
    cache = {}
    for kind in bj.INJ_TYPES:
        for severity in bj.SEVERITIES:
            rng = np.random.default_rng(1000 + fold)
            injected = dict(mean_segs)
            for uid in test_uids:
                injected[uid] = bj.inject(mean_segs[uid], kind, severity, train_sd, rng)
            injected_diff = first_difference_arrays(injected)
            cache[(kind, severity)] = (injected_diff, bj.amp_features(injected_diff, normal))
    return cache


def _injected_scores(
    ridge: RidgeVAR,
    mcd: RobustMCD,
    normal: pd.DataFrame,
    injected_inputs: dict[tuple[str, float], tuple[dict[str, np.ndarray], pd.DataFrame]],
    ridge_thr: float,
    mcd_thr: float,
    mean_thr: float,
) -> dict[str, dict[str, float]]:
    """4유형 × 4강도 합성 이상에 대한 규칙별 창 탐지율."""
    values = {rule: {} for rule in RULES}
    for kind in bj.INJ_TYPES:
        for severity in bj.SEVERITIES:
            injected_diff, amp = injected_inputs[(kind, severity)]
            ridge_score = ridge.score(injected_diff, normal)
            mcd_score = mcd.score(amp)
            rules = rule_scores(ridge_score / ridge_thr, mcd_score / mcd_thr, mean_thr)
            for rule, score in rules.items():
                values[rule][f"inj_{kind}_{severity:g}"] = float(np.mean(score > 1.0))
    return values


def _error_rows(
    rule: str,
    split_name: str,
    fold: int,
    seed: int,
    normal: pd.DataFrame,
    hit_normal: np.ndarray,
    delay: pd.DataFrame,
    attrs: pd.DataFrame,
) -> list[dict]:
    rows = []
    hit = pd.DataFrame({"seg_uid": normal["seg_uid"].to_numpy(), "hit": hit_normal})
    fp = hit.groupby("seg_uid", sort=False)["hit"].agg(["sum", "size"]).query("sum > 0")
    for uid, value in fp.iterrows():
        rows.append({
            "model": rule,
            "split": split_name,
            "fold": fold,
            "seed": seed,
            "type": "FP",
            "seg_uid": uid,
            "n_exceed": int(value["sum"]),
            "n_windows": int(value["size"]),
            "state": STATE_NAME.get(attrs.loc[uid, "state"], ""),
        })
    for uid in delay.loc[~delay["detected"], "seg_uid"]:
        rows.append({
            "model": rule,
            "split": split_name,
            "fold": fold,
            "seed": seed,
            "type": "FN",
            "seg_uid": uid,
            "vib_grade": attrs.loc[uid, "vib_grade"],
            "cur_grade": attrs.loc[uid, "cur_grade"],
        })
    return rows


def run(
    nb: str,
    seeds: tuple[int, ...] = SEEDS,
    lookback: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ridge + MCD 단독·결합 규칙을 9분할·MCD 3시드로 검증한다.

    반환: cv, window scores, segment alarms, model details, errors, timing.
    """
    _, mean_segs, diff_segs, wins = load_inputs()
    diff_amp = bj.amp_features(diff_segs, wins)
    attrs = wins.groupby("seg_uid")[["state", "vib_grade", "cur_grade"]].first()
    model_dir = paths.MODELS / nb
    model_dir.mkdir(parents=True, exist_ok=True)

    cv_rows, score_rows, segment_rows = [], [], []
    detail_rows, error_rows, timing_rows = [], [], []
    for split_name, fold, train, test in bj.iter_splits(wins):
        train_uids = train["seg_uid"].drop_duplicates().tolist()
        fit_started = time.perf_counter()
        ridge = RidgeVAR(lookback=lookback, win=int(FS)).fit([diff_segs[uid] for uid in train_uids])
        ridge_fit_sec = time.perf_counter() - fit_started
        ridge.save(model_dir / f"ridge_diff_{split_name}_f{fold}.joblib")
        ridge_train = ridge.score(diff_segs, train)
        ridge_test = ridge.score(diff_segs, test)
        ridge_thr = ev.threshold_from_normal(ridge_train)

        y = test["label"].to_numpy(dtype=int)
        normal = test[y == 0]
        outlier = test[y == 1]
        normal_mask = y == 0
        injected_inputs = _injected_inputs(mean_segs, normal, train_uids, fold)
        for seed in seeds:
            mcd_started = time.perf_counter()
            mcd = RobustMCD(bj.AMP_COLS, seed=seed).fit(diff_amp.loc[train.index])
            mcd_fit_sec = time.perf_counter() - mcd_started
            joblib.dump(mcd, model_dir / f"mcd_diff_amp_{split_name}_f{fold}_s{seed}.joblib")
            score_started = time.perf_counter()
            mcd_train = mcd.score(diff_amp.loc[train.index])
            mcd_test = mcd.score(diff_amp.loc[test.index])
            mcd_thr = ev.threshold_from_normal(mcd_train)
            z_ridge_train = ridge_train / ridge_thr
            z_mcd_train = mcd_train / mcd_thr
            mean_raw_train = 0.5 * (z_ridge_train + z_mcd_train)
            mean_thr = ev.threshold_from_normal(mean_raw_train)
            test_rules = rule_scores(ridge_test / ridge_thr, mcd_test / mcd_thr, mean_thr)
            score_sec = time.perf_counter() - score_started
            injection = _injected_scores(
                ridge, mcd, normal, injected_inputs, ridge_thr, mcd_thr, mean_thr
            )

            detail_rows.append({
                "split": split_name,
                "fold": fold,
                "seed": seed,
                "ridge_alpha": ridge.alpha,
                "ridge_threshold": ridge_thr,
                "mcd_threshold": mcd_thr,
                "mean_fusion_threshold": mean_thr,
                **mcd.details(),
            })
            timing_rows.append({
                "split": split_name,
                "fold": fold,
                "seed": seed,
                "ridge_fit_sec": ridge_fit_sec,
                "mcd_fit_sec": mcd_fit_sec,
                "test_score_sec": score_sec,
                "n_test_windows": len(test),
                "score_us_per_window": score_sec / len(test) * 1e6,
            })

            score_table = test[["seg_uid", "t_start", "label", "state"]].reset_index(drop=True).copy()
            score_table.insert(0, "seed", seed)
            score_table.insert(0, "fold", fold)
            score_table.insert(0, "split", split_name)
            score_table["z_ridge"] = ridge_test / ridge_thr
            score_table["z_mcd"] = mcd_test / mcd_thr
            for rule, score in test_rules.items():
                score_table[f"z_{rule}"] = score
            score_rows.append(score_table)

            normal_scores = score_table[normal_mask].reset_index(drop=True)
            for uid, group in normal_scores.groupby("seg_uid", sort=False):
                state = group["state"].iloc[0]
                row = {"split": split_name, "fold": fold, "seed": seed, "seg_uid": uid, "state": state}
                for rule in RULES:
                    row[rule] = bool((group[f"z_{rule}"] > 1).any())
                segment_rows.append(row)

            for rule, score in test_rules.items():
                hit = score > 1.0
                normal_score = score[normal_mask]
                outlier_score = score[~normal_mask]
                hit_normal = hit[normal_mask]
                delay = ev.detection_delay_s(
                    outlier_score,
                    outlier["seg_uid"],
                    outlier["t_start"],
                    1.0,
                    win_s=1.0,
                )
                detected = delay["detected"].to_numpy(dtype=bool)
                injected = injection[rule]
                row = {
                    "model": rule,
                    "model_name": RULE_NAME[rule],
                    "split": split_name,
                    "fold": fold,
                    "seed": seed,
                    "auc": ev.auc(y, score),
                    "fpr_sample": float(hit_normal.mean()),
                    "fpr_segment": ev.fpr_segment(normal_score, normal["seg_uid"], 1.0),
                    "delay_median_s": float(np.nanmedian(delay["delay_in_window_s"])) if detected.any() else np.nan,
                    "n_out_seg": len(delay),
                    "n_detected": int(detected.sum()),
                    "recall_window": float(hit[~normal_mask].mean()),
                    **_state_fpr(normal, hit_normal),
                    **injected,
                }
                row["inj_mean"] = float(np.mean(list(injected.values())))
                for kind in bj.INJ_TYPES:
                    row[f"inj_{kind}"] = float(np.mean([
                        injected[f"inj_{kind}_{severity:g}"] for severity in bj.SEVERITIES
                    ]))
                for severity in bj.SEVERITIES:
                    row[f"inj_sev{severity:g}"] = float(np.mean([
                        injected[f"inj_{kind}_{severity:g}"] for kind in bj.INJ_TYPES
                    ]))
                cv_rows.append(row)
                error_rows.extend(_error_rows(
                    rule, split_name, fold, seed, normal, hit_normal, delay, attrs
                ))

    return (
        pd.DataFrame(cv_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(segment_rows),
        pd.DataFrame(detail_rows),
        pd.DataFrame(error_rows),
        pd.DataFrame(timing_rows),
    )


def to_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    values = ["auc", "fpr_sample", "fpr_segment", "delay_median_s"]
    per = cv.groupby(["model", "split", "fold"], sort=False)[values].mean().reset_index()
    mean = per.groupby(["model", "split"], sort=False)[values].mean().reset_index().assign(fold="mean")
    return pd.concat([per, mean], ignore_index=True)[ev.METRIC_COLS]


def summary(cv: pd.DataFrame) -> pd.DataFrame:
    values = [
        "auc", "fpr_sample", "fpr_segment", "n_detected", "n_out_seg", "recall_window",
        "delay_median_s", "fpr_sample_high_load", "fpr_sample_low_load",
        "fpr_segment_high_load", "fpr_segment_low_load", "inj_mean",
        *[f"inj_{kind}" for kind in bj.INJ_TYPES],
        *[f"inj_sev{severity:g}" for severity in bj.SEVERITIES],
    ]
    return cv.groupby(["model", "split"], sort=False)[values].mean().reset_index()


def fp_overlap(segment_alarms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in segment_alarms.groupby(["split", "fold", "seed"], sort=False):
        a = group["ridge_diff"].to_numpy(dtype=bool)
        b = group["mcd_diff_amp"].to_numpy(dtype=bool)
        intersection = int(np.sum(a & b))
        union = int(np.sum(a | b))
        rows.append({
            "split": key[0], "fold": key[1], "seed": key[2],
            "n_normal_segments": len(group),
            "ridge_fp": int(a.sum()), "mcd_fp": int(b.sum()),
            "intersection_fp": intersection, "union_fp": union,
            "jaccard": intersection / union if union else 0.0,
        })
    return pd.DataFrame(rows)


def paired_bootstrap(
    segment_alarms: pd.DataFrame,
    base: str = "ridge_diff",
    other: str = "ridge&mcd",
    n_boot: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """세그먼트 복원추출로 결합−단독 FP 차이의 95% 구간."""
    rows = []
    rng = np.random.default_rng(seed)
    for split_name, group in segment_alarms.groupby("split", sort=False):
        per_seg = group.groupby("seg_uid")[[base, other]].mean()
        delta = (per_seg[other] - per_seg[base]).to_numpy(dtype=float)
        boot = np.asarray([
            delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_boot)
        ])
        rows.append({
            "split": split_name,
            "base": base,
            "other": other,
            "n_segments": len(delta),
            "fpr_base": float(per_seg[base].mean()),
            "fpr_other": float(per_seg[other].mean()),
            "difference": float(delta.mean()),
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
        })
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e4e0",
        "axes.axisbelow": True,
    })


def plot_quadrant(scores: pd.DataFrame, fig_dir: Path, name: str = "ridge_mcd_quadrant.png") -> None:
    _style()
    data = scores[(scores["split"] == "time_block") & (scores["fold"] == 4) & (scores["seed"] == 0)]
    x = np.clip(data["z_ridge"].to_numpy(), 1e-4, None)
    y = np.clip(data["z_mcd"].to_numpy(), 1e-4, None)
    label = data["label"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for value, color, text in ((0, "#8a8984", "정상"), (1, "#d6332a", "실제 이상")):
        mask = label == value
        ax.scatter(x[mask], y[mask], s=10 if value == 0 else 18, alpha=0.5 if value == 0 else 0.8,
                   color=color, label=text, edgecolors="none")
    ax.axvline(1, color="#444444", ls="--", lw=1)
    ax.axhline(1, color="#444444", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Ridge 점수 / 정상 q99")
    ax.set_ylabel("MCD 점수 / 정상 q99")
    ax.set_title("서로 다른 관점의 정상 오경보 겹침", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)


def plot_tradeoff(cv: pd.DataFrame, fig_dir: Path, name: str = "ensemble_tradeoff.png") -> None:
    _style()
    data = cv[cv["split"] == "time_block"].groupby("model", sort=False)[["fpr_segment", "inj_sev1"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for model, row in data.iterrows():
        color = "#eb6834" if "&" in model else "#2a78d6" if model == "ridge_diff" else "#8a8984"
        ax.scatter(row["fpr_segment"], row["inj_sev1"], s=55, color=color, edgecolors="white")
        ax.annotate(model, (row["fpr_segment"], row["inj_sev1"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("세그먼트 오경보율")
    ax.set_ylabel("합성 이상 강도 1 탐지율")
    ax.set_title("오경보와 약한 이상 탐지의 맞바꿈", loc="left")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)


def plot_strength(cv: pd.DataFrame, fig_dir: Path, name: str = "synthetic_strength.png") -> None:
    _style()
    cols = [f"inj_sev{severity:g}" for severity in bj.SEVERITIES]
    data = cv[cv["split"] == "time_block"].groupby("model", sort=False)[cols].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in ("ridge_diff", "mcd_diff_amp", "ridge&mcd", "ridge+mcd[cal]"):
        ax.plot(bj.SEVERITIES, data.loc[model], marker="o", label=model)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bj.SEVERITIES, [f"{value:g}" for value in bj.SEVERITIES])
    ax.set_xlabel("합성 이상 강도")
    ax.set_ylabel("창 단위 탐지율")
    ax.set_title("합성 이상 강도별 탐지율", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)


def plot_state(cv: pd.DataFrame, fig_dir: Path, name: str = "state_fpr.png") -> None:
    _style()
    models = ["ridge_diff", "mcd_diff_amp", "ridge&mcd", "ridge+mcd[cal]"]
    data = cv[cv["split"] == "time_block"].groupby("model", sort=False)[
        ["fpr_segment_high_load", "fpr_segment_low_load"]
    ].mean().loc[models]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, data["fpr_segment_high_load"], 0.38, color="#eb6834", label="고부하")
    ax.bar(x + 0.2, data["fpr_segment_low_load"], 0.38, color="#2a78d6", label="저부하")
    ax.set_xticks(x, data.index, rotation=22, ha="right")
    ax.set_ylabel("정상 세그먼트 오경보율")
    ax.set_title("운전 상태별 오경보", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)
