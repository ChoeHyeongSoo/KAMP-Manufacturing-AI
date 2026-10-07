"""34번 Ridge-VAR 실시간 입력 검증.

오프라인 세그먼트 평균 제거와 두 causal 입력 후보를 같은 분할·임계 계약으로 비교한다.

- ``offline_mean``: 세그먼트 전체 평균 제거. 22번 결과 재현용이며 실시간 입력이 아니다.
- ``fixed_baseline``: 학습 정상 긴 세그먼트에서 구한 채널별 상수를 뺀다.
- ``first_difference``: 세그먼트 안 1차 차분. 첫 샘플은 0이며 DC 상수에 불변이다.

모든 입력은 먼저 공통 형식 통일을 거친다. Ridge는 burst마다 과거 버퍼를 비우며 처음
``lookback=5``개 샘플에는 점수를 내지 않는다. 임계값은 fold별 학습 정상 점수 q99다.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import benchmark_jsc as bj
import data_quality as dq
import evaluate as ev
from models_jsc import FS, RidgeVAR
import paths
import preprocess

VARIANTS = ("offline_mean", "fixed_baseline", "first_difference")
VARIANT_NAME = {
    "offline_mean": "Offline mean",
    "fixed_baseline": "Fixed baseline",
    "first_difference": "First difference",
}
STATE_NAME = {0: "고부하", 1: "저부하"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """원본·형식 통일 샘플과 1초 윈도우 계약을 읽는다."""
    raw = pd.concat(dq.load_all().values(), ignore_index=True)
    raw = preprocess.add_seg_uid(raw)
    unified = preprocess.unify_format(raw)
    wins = pd.read_parquet(paths.DATA_PROCESSED / "11_window_features.parquet")
    wins = wins[wins["win_s"] == 1.0].reset_index(drop=True)
    return raw, unified, wins


def segment_arrays(frame: pd.DataFrame, dtype=np.float32) -> dict[str, np.ndarray]:
    """샘플 프레임을 시간순 ``seg_uid -> (n, 3)`` 배열로 바꾼다."""
    return {
        uid: group[preprocess.SENSORS].to_numpy(dtype)
        for uid, group in frame.groupby("seg_uid", sort=False)
    }


def prepare_variant(
    unified: pd.DataFrame,
    variant: str,
    train_uids: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float] | None]:
    """형식 통일 샘플에 입력 변형을 적용한다.

    ``fixed_baseline``의 상수는 반드시 현재 fold의 학습 정상 UID만으로 적합한다.
    ``first_difference``는 세그먼트 경계를 넘지 않고 첫 샘플을 0으로 둔다. 첫 5샘플은
    Ridge cold-start라 점수에 쓰이지 않으므로 이 0은 실제 판정값이 되지 않는다.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant는 {VARIANTS} 중 하나: {variant!r}")
    baseline = None
    if variant == "offline_mean":
        out = preprocess.remove_dc(unified, method="mean")
    elif variant == "fixed_baseline":
        if not train_uids:
            raise ValueError("fixed_baseline에는 현재 fold의 train_uids가 필요하다")
        train = unified[unified["seg_uid"].isin(train_uids)]
        baseline = preprocess.fit_baseline(train)
        out = preprocess.remove_dc(unified, method="baseline", baseline=baseline)
    else:
        out = unified.copy()
        diff = out.groupby("seg_uid", sort=False)[preprocess.SENSORS].diff()
        out[preprocess.SENSORS] = diff.fillna(0.0)
    return segment_arrays(out), baseline


def _stream_prepared(model: RidgeVAR, x_prepared: np.ndarray) -> np.ndarray:
    """이미 변형된 한 burst를 샘플 하나씩 처리한다."""
    x = np.asarray(x_prepared, dtype=np.float64)
    z = (x - model.mu) / model.sd
    score = np.full(len(z), np.nan)
    history: list[np.ndarray] = []
    for t, sample in enumerate(z):
        if len(history) == model.lookback:
            xx = np.asarray(history, dtype=np.float64).reshape(1, -1)
            pred = model.model.predict(xx)[0]
            err = (pred - sample) / model.err_sd
            score[t] = float(np.mean(err ** 2))
            history.pop(0)
        history.append(sample)
    return score


def stream_scores(
    model: RidgeVAR,
    x_unified: np.ndarray,
    variant: str,
    baseline: dict[str, float] | None,
    x_offline_mean: np.ndarray | None = None,
) -> np.ndarray:
    """burst 경계마다 상태를 초기화하는 단일 샘플 입력 경로.

    공통 형식 통일까지 끝난 센서값을 받는다고 가정한다. causal 후보는 각 샘플 도착 시
    상수 제거 또는 직전 샘플 차분을 수행한다. ``offline_mean``만 비교 기준이라 완성된
    burst의 평균 제거 배열을 받는다.
    """
    if variant == "offline_mean":
        if x_offline_mean is None:
            raise ValueError("offline_mean에는 완성 burst의 x_offline_mean이 필요하다")
        return _stream_prepared(model, x_offline_mean)

    x = np.asarray(x_unified, dtype=np.float64)
    base = None
    if variant == "fixed_baseline":
        if baseline is None:
            raise ValueError("fixed_baseline에는 baseline이 필요하다")
        base = np.asarray([baseline[ch] for ch in preprocess.SENSORS], dtype=np.float64)

    score = np.full(len(x), np.nan)
    history: list[np.ndarray] = []
    prev = None
    for t, sample in enumerate(x):
        if variant == "fixed_baseline":
            transformed = sample - base
        elif variant == "first_difference":
            transformed = np.zeros_like(sample) if prev is None else sample - prev
            prev = sample.copy()
        else:
            raise ValueError(variant)
        # prepare_variant가 저장하는 float32 입력과 완전히 같은 수치 경로를 사용한다.
        transformed = np.asarray(transformed, dtype=np.float32).astype(np.float64)
        z = (transformed - model.mu) / model.sd
        if len(history) == model.lookback:
            xx = np.asarray(history, dtype=np.float64).reshape(1, -1)
            pred = model.model.predict(xx)[0]
            err = (pred - z) / model.err_sd
            score[t] = float(np.mean(err ** 2))
            history.pop(0)
        history.append(z)
    return score


def _state_fpr(normal: pd.DataFrame, score: np.ndarray, threshold: float) -> dict[str, float]:
    """고·저부하별 윈도우 및 세그먼트 오경보율."""
    table = normal[["seg_uid", "state"]].copy()
    table["hit"] = np.asarray(score) > threshold
    out = {}
    for state, key in ((0, "high_load"), (1, "low_load")):
        part = table[table["state"] == state]
        out[f"fpr_sample_{key}"] = float(part["hit"].mean()) if len(part) else np.nan
        per_seg = part.groupby("seg_uid", sort=False)["hit"].any()
        out[f"fpr_segment_{key}"] = float(per_seg.mean()) if len(per_seg) else np.nan
    return out


def _dc_diagnostic(
    segs: dict[str, np.ndarray], wins: pd.DataFrame, variant: str, split_name: str, fold: int
) -> list[dict]:
    """잔여 세그먼트 평균만으로 라벨이 분리되는지 보는 진단(모델 입력 선택에는 미사용)."""
    label = wins.groupby("seg_uid", sort=False)["label"].first()
    uids = [uid for uid in label.index if uid in segs]
    y = label.loc[uids].to_numpy(dtype=int)
    rows = []
    for index, channel in enumerate(preprocess.SENSORS):
        value = np.asarray([abs(float(np.mean(segs[uid][:, index]))) for uid in uids])
        # mean 제거 후 float32 합산에서 생기는 1e-6 수준 잔차를 라벨 신호로 오인하지 않는다.
        value[value < 1e-5] = 0.0
        rows.append({
            "variant": variant,
            "split": split_name,
            "fold": fold,
            "channel": channel,
            "abs_segment_mean_auc": float(roc_auc_score(y, value)),
        })
    return rows


def _benchmark_stream(
    model: RidgeVAR,
    unified_segs: dict[str, np.ndarray],
    prepared_segs: dict[str, np.ndarray],
    uids: list[str],
    variant: str,
    baseline: dict[str, float] | None,
    repeats: int = 3,
) -> tuple[dict[str, float], float, float]:
    """단일 샘플 입력 경로의 처리량과 batch/stream 점수 정합성을 잰다."""
    use = list(dict.fromkeys(uids))
    received = int(sum(len(unified_segs[uid]) for uid in use))
    scored = int(sum(max(0, len(unified_segs[uid]) - model.lookback) for uid in use))
    elapsed = []
    max_error = 0.0
    max_relative_error = 0.0
    for repeat in range(repeats):
        started = time.perf_counter_ns()
        for uid in use:
            got = stream_scores(
                model,
                unified_segs[uid],
                variant,
                baseline,
                x_offline_mean=prepared_segs[uid] if variant == "offline_mean" else None,
            )
            if repeat == 0:
                expected = model.step_scores(prepared_segs[uid])
                finite = np.isfinite(expected) & np.isfinite(got)
                if finite.any():
                    absolute = np.abs(expected[finite] - got[finite])
                    relative = absolute / np.maximum(np.abs(expected[finite]), 1.0)
                    max_error = max(max_error, float(np.max(absolute)))
                    max_relative_error = max(max_relative_error, float(np.max(relative)))
                if not np.array_equal(np.isnan(expected), np.isnan(got)):
                    raise AssertionError(f"batch/stream NaN 위치 불일치: {variant}, {uid}")
        elapsed.append((time.perf_counter_ns() - started) / 1e9)
    sec = float(np.median(elapsed))
    return {
        "stream_sec": sec,
        "n_received_samples": received,
        "n_scored_samples": scored,
        "us_per_received_sample": sec / received * 1e6 if received else np.nan,
        "us_per_scored_sample": sec / scored * 1e6 if scored else np.nan,
        "throughput_received_hz": received / sec if sec else np.nan,
    }, max_error, max_relative_error


def _error_rows(
    variant: str,
    split_name: str,
    fold: int,
    normal: pd.DataFrame,
    score_normal: np.ndarray,
    delay: pd.DataFrame,
    threshold: float,
    attrs: pd.DataFrame,
) -> list[dict]:
    rows = []
    hit = pd.DataFrame({"seg_uid": normal["seg_uid"].to_numpy(), "hit": score_normal > threshold})
    fp = hit.groupby("seg_uid", sort=False)["hit"].agg(["sum", "size"]).query("sum > 0")
    for uid, value in fp.iterrows():
        rows.append({
            "variant": variant,
            "split": split_name,
            "fold": fold,
            "type": "FP",
            "seg_uid": uid,
            "n_exceed": int(value["sum"]),
            "n_windows": int(value["size"]),
            "state": STATE_NAME.get(attrs.loc[uid, "state"], ""),
        })
    for uid in delay.loc[~delay["detected"], "seg_uid"]:
        rows.append({
            "variant": variant,
            "split": split_name,
            "fold": fold,
            "type": "FN",
            "seg_uid": uid,
            "vib_grade": attrs.loc[uid, "vib_grade"],
            "cur_grade": attrs.loc[uid, "cur_grade"],
        })
    return rows


def run(
    nb: str,
    seed: int = 0,
    lookback: int = 5,
    repeats: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """세 입력 × 9분할 Ridge 검증 전체 실행.

    반환: cv, window scores, timing, fold baseline, input diagnostics, errors, contract checks.
    """
    _, unified, wins = load_inputs()
    # 온라인 경로는 형식 통일 직후의 float64에서 입력 변형 후 모델 계약(float32)으로 내린다.
    # pandas 일괄 전처리와 같은 연산 순서를 보존해 batch/stream 정합성을 직접 검증한다.
    unified_segs = segment_arrays(unified, dtype=np.float64)
    mean_segs, _ = prepare_variant(unified, "offline_mean")
    diff_segs, _ = prepare_variant(unified, "first_difference")
    fixed = {"offline_mean": mean_segs, "first_difference": diff_segs}
    attrs = wins.groupby("seg_uid")[["state", "vib_grade", "cur_grade"]].first()

    cv_rows, score_rows, timing_rows = [], [], []
    baseline_rows, diagnostic_rows, error_rows, check_rows = [], [], [], []
    model_dir = paths.MODELS / nb
    model_dir.mkdir(parents=True, exist_ok=True)

    for split_name, fold, train, test in bj.iter_splits(wins):
        train_uids = train["seg_uid"].drop_duplicates().tolist()
        test_uids = test["seg_uid"].drop_duplicates().tolist()
        for variant in VARIANTS:
            prep_started = time.perf_counter()
            if variant == "fixed_baseline":
                segs, baseline = prepare_variant(unified, variant, train_uids)
            else:
                segs, baseline = fixed[variant], None
            preprocess_sec = time.perf_counter() - prep_started

            if baseline is not None:
                baseline_rows.append({
                    "split": split_name,
                    "fold": fold,
                    **{f"baseline_{channel}": value for channel, value in baseline.items()},
                })
            diagnostic_rows.extend(_dc_diagnostic(segs, wins, variant, split_name, fold))

            model = RidgeVAR(lookback=lookback, win=int(FS))
            fit_started = time.perf_counter()
            model.fit([segs[uid] for uid in train_uids], seed=seed)
            fit_sec = time.perf_counter() - fit_started
            model.save(model_dir / f"ridge_{variant}_{split_name}_f{fold}_s{seed}.joblib")

            score_started = time.perf_counter()
            train_score = model.score(segs, train)
            test_score = model.score(segs, test)
            batch_score_sec = time.perf_counter() - score_started
            threshold = ev.threshold_from_normal(train_score)
            y = test["label"].to_numpy(dtype=int)
            normal = test[y == 0]
            outlier = test[y == 1]
            score_normal = test_score[y == 0]
            score_outlier = test_score[y == 1]
            delay = ev.detection_delay_s(
                score_outlier,
                outlier["seg_uid"],
                outlier["t_start"],
                threshold,
                win_s=1.0,
            )
            detected = delay["detected"].to_numpy(dtype=bool)
            state_fpr = _state_fpr(normal, score_normal, threshold)
            row = {
                "model": f"Ridge-VAR [{VARIANT_NAME[variant]}]",
                "variant": variant,
                "split": split_name,
                "fold": fold,
                "seed": seed,
                "auc": ev.auc(y, test_score),
                "fpr_sample": ev.fpr_sample(score_normal, threshold),
                "fpr_segment": ev.fpr_segment(score_normal, normal["seg_uid"], threshold),
                "delay_median_s": float(np.nanmedian(delay["delay_in_window_s"])) if detected.any() else np.nan,
                "felt_delay_median_s": (
                    float(np.nanmedian(delay["delay_in_window_s"])) + ev.BURST_GAP_S
                    if detected.any() else np.nan
                ),
                "n_out_seg": len(delay),
                "n_detected": int(detected.sum()),
                "recall_window": float(np.mean(score_outlier > threshold)),
                "thr": threshold,
                "alpha": model.alpha,
                "fit_sec": fit_sec,
                "preprocess_sec": preprocess_sec,
                "batch_score_sec": batch_score_sec,
                "first_score_s": lookback / FS,
                "first_window_decision_s": (int(FS) - 1) / FS,
                **state_fpr,
            }
            cv_rows.append(row)

            timing, max_error, max_relative_error = _benchmark_stream(
                model, unified_segs, segs, test_uids, variant, baseline, repeats=repeats
            )
            timing_rows.append({"variant": variant, "split": split_name, "fold": fold, **timing})
            first_finite = []
            reset_ok = True
            for uid in test_uids:
                value = model.step_scores(segs[uid])
                pos = np.flatnonzero(np.isfinite(value))
                if len(pos):
                    first_finite.append(int(pos[0]))
                reset_ok &= bool(np.isnan(value[: min(lookback, len(value))]).all())
            check_rows.append({
                "variant": variant,
                "split": split_name,
                "fold": fold,
                "boundary_reset_ok": reset_ok,
                "first_finite_index_min": min(first_finite) if first_finite else np.nan,
                "first_finite_index_max": max(first_finite) if first_finite else np.nan,
                "batch_stream_max_abs_error": max_error,
                "batch_stream_max_relative_error": max_relative_error,
            })

            error_rows.extend(
                _error_rows(
                    variant, split_name, fold, normal, score_normal, delay, threshold, attrs
                )
            )
            score_rows.append(pd.DataFrame({
                "variant": variant,
                "split": split_name,
                "fold": fold,
                "seg_uid": test["seg_uid"].to_numpy(),
                "label": y,
                "state": test["state"].to_numpy(),
                "t_start": test["t_start"].to_numpy(),
                "score": test_score,
                "thr": threshold,
            }))

    return (
        pd.DataFrame(cv_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(timing_rows),
        pd.DataFrame(baseline_rows),
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(error_rows),
        pd.DataFrame(check_rows),
    )


def to_metrics(cv: pd.DataFrame) -> pd.DataFrame:
    """공통 metrics 형식의 fold 및 평균 행."""
    values = ["auc", "fpr_sample", "fpr_segment", "delay_median_s"]
    per = cv[["model", "split", "fold", *values]].copy()
    mean = cv.groupby(["model", "split"], sort=False)[values].mean().reset_index().assign(fold="mean")
    return pd.concat([per, mean], ignore_index=True)[ev.METRIC_COLS]


def summary(cv: pd.DataFrame) -> pd.DataFrame:
    values = [
        "auc", "fpr_sample", "fpr_segment", "n_detected", "n_out_seg", "delay_median_s",
        "fpr_sample_high_load", "fpr_sample_low_load", "fpr_segment_high_load",
        "fpr_segment_low_load", "fit_sec", "batch_score_sec",
    ]
    return cv.groupby(["variant", "split"], sort=False)[values].mean().reset_index()


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


def plot_comparison(cv: pd.DataFrame, fig_dir: Path, name: str = "input_comparison.png") -> None:
    """입력별 주요 성능(time-block fold 평균)."""
    _style()
    data = cv[cv["split"] == "time_block"].groupby("variant", sort=False).agg(
        auc=("auc", "mean"),
        fpr_segment=("fpr_segment", "mean"),
        n_detected=("n_detected", "mean"),
        delay_median_s=("delay_median_s", "mean"),
    ).reindex(VARIANTS)
    labels = [VARIANT_NAME[v] for v in data.index]
    colors = ["#9a9993", "#eda100", "#2a78d6"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (column, title) in zip(axes, (
        ("auc", "AUC"), ("fpr_segment", "세그먼트 오경보율"),
        ("n_detected", "이상 탐지 수"), ("delay_median_s", "탐지 지연(초)"),
    )):
        ax.bar(labels, data[column], color=colors, edgecolor="white")
        ax.set_title(title, loc="left", fontsize=10)
        ax.tick_params(axis="x", rotation=22, labelsize=8)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)


def plot_state_fpr(cv: pd.DataFrame, fig_dir: Path, name: str = "state_fpr.png") -> None:
    """causal 후보의 운전 상태별 세그먼트 오경보(time-block)."""
    _style()
    data = cv[cv["split"] == "time_block"].groupby("variant", sort=False)[
        ["fpr_segment_high_load", "fpr_segment_low_load"]
    ].mean().reindex(VARIANTS)
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, data["fpr_segment_high_load"], 0.38, label="고부하", color="#eb6834")
    ax.bar(x + 0.2, data["fpr_segment_low_load"], 0.38, label="저부하", color="#2a78d6")
    ax.set_xticks(x, [VARIANT_NAME[v] for v in data.index])
    ax.set_ylabel("정상 세그먼트 오경보율")
    ax.set_title("입력별 운전 상태 오경보", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)


def plot_timing(timing: pd.DataFrame, fig_dir: Path, name: str = "stream_timing.png") -> None:
    """단일 샘플 스트리밍 처리시간(time-block 중앙값)."""
    _style()
    data = timing[timing["split"] == "time_block"].groupby("variant", sort=False)[
        "us_per_received_sample"
    ].median().reindex(VARIANTS)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([VARIANT_NAME[v] for v in data.index], data, color=["#9a9993", "#eda100", "#2a78d6"])
    ax.set_ylabel("수신 샘플당 처리시간 (µs)")
    ax.set_title("Ridge 단일 샘플 스트리밍 처리시간", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / name, dpi=120)
    plt.close(fig)
