"""Ridge·MCD 최종 2단계 경보의 정적 HTML 대시보드 생성.

23번의 저장된 교차검증 점수와 세그먼트 메타데이터만 읽는다. 모델을 다시
학습하지 않으며, 외부 서버·계정·JavaScript 라이브러리 없이 단일 HTML을
만든다. 화면은 실제 센서 연결이 아니라 저장 데이터의 스트림 재생이다.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths


MODEL_NB = "23_model_ensemble_ridge_mcd_JSC"
REALTIME_NB = "34_realtime_input_validation_JSC"


def load_inputs() -> dict[str, pd.DataFrame]:
    """대시보드 입력을 읽고 필요한 열을 검증한다."""
    model_dir = paths.RESULTS / MODEL_NB
    realtime_dir = paths.RESULTS / REALTIME_NB
    files = {
        "scores": model_dir / "scores.csv",
        "summary": model_dir / "summary.csv",
        "timing": model_dir / "timing.csv",
        "segments": paths.DATA_PROCESSED / "segments.csv",
        "stream_timing": realtime_dir / "stream_timing.csv",
    }
    missing = [str(p.relative_to(paths.ROOT)) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("대시보드 입력이 없습니다: " + ", ".join(missing))

    data = {name: pd.read_csv(path, encoding="utf-8-sig") for name, path in files.items()}
    required_scores = {
        "split", "fold", "seed", "seg_uid", "t_start", "label", "state",
        "z_ridge_diff", "z_mcd_diff_amp", "z_ridge&mcd",
    }
    required_segments = {"seg_uid", "start", "state"}
    if missing_cols := required_scores - set(data["scores"].columns):
        raise ValueError(f"scores.csv 필수 열 누락: {sorted(missing_cols)}")
    if missing_cols := required_segments - set(data["segments"].columns):
        raise ValueError(f"segments.csv 필수 열 누락: {sorted(missing_cols)}")
    return data


def prepare_replay(
    scores: pd.DataFrame,
    segments: pd.DataFrame,
    split: str = "time_block",
    seed: int = 0,
    outlier_fold: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정상 평가 열과 대표 이상 이벤트를 창·세그먼트 표로 만든다.

    정상은 time-block의 평가 블록 1~4를 이어 붙인다. 이때 정상 세그먼트는
    fold 사이에 중복되지 않는다. 이상은 같은 이벤트가 모든 fold에 반복
    평가되므로 가장 많은 과거 정상으로 학습한 fold 4·seed 0 한 벌만 표시한다.
    """
    d = scores[(scores["split"] == split) & (scores["seed"] == seed)].copy()
    normal = d[d["label"] == 0].copy()
    outlier = d[(d["label"] == 1) & (d["fold"] == outlier_fold)].copy()
    normal["view"] = "normal"
    outlier["view"] = "outlier"
    replay = pd.concat([normal, outlier], ignore_index=True)

    meta = segments[["seg_uid", "start", "state"]].rename(columns={"state": "state_meta"})
    replay = replay.merge(meta, on="seg_uid", how="left", validate="many_to_one")
    if replay["start"].isna().any():
        missing = replay.loc[replay["start"].isna(), "seg_uid"].unique().tolist()
        raise ValueError(f"segments.csv에서 시작 시각을 찾지 못함: {missing[:5]}")
    replay["state"] = replay["state"].fillna(replay["state_meta"])
    replay["abs_time"] = pd.to_datetime(replay["start"]) + pd.to_timedelta(replay["t_start"], unit="s")
    replay["relative_sec"] = replay.groupby("view")["abs_time"].transform(
        lambda s: (s - s.min()).dt.total_seconds()
    )
    replay["stage"] = np.select(
        [replay["z_ridge&mcd"] > 1, replay["z_ridge_diff"] > 1],
        ["alarm", "caution"],
        default="normal",
    )

    group_cols = ["view", "split", "fold", "seed", "seg_uid"]
    segment = replay.groupby(group_cols, as_index=False, sort=False).agg(
        relative_sec=("relative_sec", "min"),
        label=("label", "first"),
        state=("state", "first"),
        ridge_max=("z_ridge_diff", "max"),
        mcd_max=("z_mcd_diff_amp", "max"),
        alarm_max=("z_ridge&mcd", "max"),
        n_windows=("t_start", "size"),
    )
    segment["stage"] = np.select(
        [segment["alarm_max"] > 1, segment["ridge_max"] > 1],
        ["alarm", "caution"],
        default="normal",
    )
    state_map = {0.0: "고부하", 1.0: "저부하"}
    segment["state_name"] = segment["state"].map(state_map).fillna("미분류")
    return replay.sort_values(["view", "abs_time"]).reset_index(drop=True), segment.sort_values(
        ["view", "relative_sec"]
    ).reset_index(drop=True)


def dashboard_metrics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """검증 표에서 화면 카드에 사용할 수치를 한 번만 추출한다."""
    tb = data["summary"][data["summary"]["split"] == "time_block"].set_index("model")
    ridge = tb.loc["ridge_diff"]
    alarm = tb.loc["ridge&mcd"]
    stream = data["stream_timing"]
    stream = stream[stream["variant"] == "first_difference"]
    timing = data["timing"]
    rows = [
        ("주의 세그먼트 FPR", ridge.fpr_segment, "ratio", "23 summary.csv", "time-block 평균"),
        ("경보 세그먼트 FPR", alarm.fpr_segment, "ratio", "23 summary.csv", "time-block 평균"),
        ("주의 이상 탐지", ridge.n_detected, "segments", "23 summary.csv", "판정 가능 17세그먼트"),
        ("경보 이상 탐지", alarm.n_detected, "segments", "23 summary.csv", "판정 가능 17세그먼트"),
        ("경보 지연 중앙값", alarm.delay_median_s, "s", "23 summary.csv", "burst 내부 최초 판정"),
        ("경보 고부하 FPR", alarm.fpr_segment_high_load, "ratio", "23 summary.csv", "time-block"),
        ("경보 저부하 FPR", alarm.fpr_segment_low_load, "ratio", "23 summary.csv", "time-block"),
        (
            "Ridge 샘플 처리 최댓값",
            stream.us_per_received_sample.max() / 1000,
            "ms/sample",
            "34 stream_timing.csv",
            "형식 통일 후 입력 변환·예측, I/O 제외",
        ),
        (
            "MCD 포함 점수 계산 최댓값",
            timing.score_us_per_window.max() / 1000,
            "ms/window",
            "23 timing.csv",
            "피처 준비 후 점수 계산, I/O 제외",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "source", "note"])


def validate_contract(replay: pd.DataFrame, segment: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """표시 데이터가 23번 최종 규칙과 같은지 검증하고 점검표를 반환한다."""
    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "AND 점수 관계",
        np.allclose(replay["z_ridge&mcd"], np.minimum(replay["z_ridge_diff"], replay["z_mcd_diff_amp"])),
        "z_alarm = min(z_ridge, z_mcd)",
    ))
    normal = segment[segment["view"] == "normal"]
    outlier = segment[segment["view"] == "outlier"]
    checks.extend([
        ("정상 평가 세그먼트", normal["seg_uid"].nunique() == 417, f"{normal['seg_uid'].nunique()}개"),
        ("대표 이상 세그먼트", outlier["seg_uid"].nunique() == 17, f"{outlier['seg_uid'].nunique()}개"),
        ("이상 경보 포함", bool((outlier["stage"] == "alarm").all()), f"경보 {(outlier['stage'] == 'alarm').sum()}/17"),
        ("정상 저부하 경보 없음", not bool(((normal["state"] == 1) & (normal["stage"] == "alarm")).any()), "저부하 경보 0개"),
        (
            "저장 성능표 탐지 일치",
            metrics.loc[metrics.metric == "경보 이상 탐지", "value"].iloc[0] == 17,
            "time-block 17/17",
        ),
    ])
    out = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    if not out["passed"].all():
        failed = out.loc[~out["passed"], "check"].tolist()
        raise AssertionError("대시보드 계약 검증 실패: " + ", ".join(failed))
    return out


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#f7f9fc",
        "axes.facecolor": "white",
    })


def plot_normal_replay(segment: pd.DataFrame, fig_dir: Path, name: str = "normal_replay.png") -> None:
    """정상 time-block 평가 열의 점수·운전 상태·경보를 그린다."""
    _style()
    d = segment[segment["view"] == "normal"].copy()
    x = d["relative_sec"].to_numpy() / 60
    fig, (ax, state_ax) = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax.plot(x, np.maximum(d["ridge_max"], 1e-4), color="#2f6fed", lw=1.2, label="Ridge caution score")
    ax.plot(x, np.maximum(d["mcd_max"], 1e-4), color="#00a896", lw=1.0, alpha=0.8, label="MCD confirm score")
    ax.plot(x, np.maximum(d["alarm_max"], 1e-4), color="#e4572e", lw=1.2, label="AND alarm score")
    ax.axhline(1, color="#252b36", lw=1, ls="--", label="threshold = 1")
    flagged = d[d["stage"] != "normal"]
    colors = flagged["stage"].map({"caution": "#f5a623", "alarm": "#d7263d"})
    ax.scatter(flagged["relative_sec"] / 60, np.maximum(flagged["ridge_max"], 1e-4), c=colors, s=34, zorder=4)
    ax.set_yscale("log")
    ax.set_ylabel("normalized score (log)")
    ax.set_title("Normal stream replay — time-block test blocks 1–4, seed 0", loc="left", weight="bold")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    state_y = np.where(d["state"].to_numpy() == 0, 1, 0)
    stage_colors = d["stage"].map({"normal": "#9aa4b2", "caution": "#f5a623", "alarm": "#d7263d"})
    state_ax.scatter(x, state_y, c=stage_colors, marker="s", s=13)
    state_ax.set_yticks([0, 1], ["Low", "High"])
    state_ax.set_ylim(-0.6, 1.6)
    state_ax.set_ylabel("load")
    state_ax.set_xlabel("elapsed time (min)")
    state_ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)


def plot_outlier_replay(replay: pd.DataFrame, fig_dir: Path, name: str = "outlier_replay.png") -> None:
    """대표 이상 이벤트의 창별 Ridge·MCD·AND 점수를 그린다."""
    _style()
    d = replay[replay["view"] == "outlier"].copy()
    x = d["relative_sec"].to_numpy()
    fig, (ax, stage_ax) = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    ax.plot(x, np.maximum(d["z_ridge_diff"], 1e-4), color="#2f6fed", lw=1.4, marker="o", ms=2.5, label="Ridge caution score")
    ax.plot(x, np.maximum(d["z_mcd_diff_amp"], 1e-4), color="#00a896", lw=1.2, marker="o", ms=2.2, label="MCD confirm score")
    ax.plot(x, np.maximum(d["z_ridge&mcd"], 1e-4), color="#e4572e", lw=1.5, label="AND alarm score")
    ax.axhline(1, color="#252b36", lw=1, ls="--", label="threshold = 1")
    ax.set_yscale("log")
    ax.set_ylabel("normalized score (log)")
    ax.set_title("Abnormal event — forward fold 4, seed 0", loc="left", weight="bold")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    stage_code = d["stage"].map({"normal": 0, "caution": 1, "alarm": 2})
    stage_colors = d["stage"].map({"normal": "#9aa4b2", "caution": "#f5a623", "alarm": "#d7263d"})
    stage_ax.scatter(x, stage_code, c=stage_colors, marker="s", s=22)
    stage_ax.set_yticks([0, 1, 2], ["Normal", "Caution", "Alarm"])
    stage_ax.set_ylim(-0.5, 2.5)
    stage_ax.set_ylabel("stage")
    stage_ax.set_xlabel("elapsed time (s)")
    stage_ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)


def plot_preview(metrics: pd.DataFrame, fig_dir: Path, name: str = "dashboard_preview.png") -> None:
    """보고서에서 바로 인용할 최종 운영점 요약 그림."""
    _style()
    m = metrics.set_index("metric")["value"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(["Ridge\ncaution", "Ridge AND MCD\nalarm"],
                [100 * m["주의 세그먼트 FPR"], 100 * m["경보 세그먼트 FPR"]],
                color=["#f5a623", "#d7263d"], width=0.58)
    axes[0].set_ylabel("segment false-positive rate (%)")
    axes[0].set_title("Forward time-block operation point", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.2)
    for i, v in enumerate([m["주의 세그먼트 FPR"], m["경보 세그먼트 FPR"]]):
        axes[0].text(i, 100 * v + 0.05, f"{100*v:.2f}%", ha="center", weight="bold")

    axes[1].bar(["High load", "Low load"],
                [100 * m["경보 고부하 FPR"], 100 * m["경보 저부하 FPR"]],
                color=["#d7263d", "#2f6fed"], width=0.58)
    axes[1].set_ylabel("alarm segment false-positive rate (%)")
    axes[1].set_title("Alarm false positives by operating state", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    for i, v in enumerate([m["경보 고부하 FPR"], m["경보 저부하 FPR"]]):
        axes[1].text(i, 100 * v + 0.015, f"{100*v:.2f}%", ha="center", weight="bold")
    fig.suptitle("Ridge caution → MCD confirmation | detected 17/17 evaluable segments | median delay 0.9 s",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=130)
    plt.close(fig)


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="data", escape=True)


def write_dashboard(
    fig_dir: Path,
    out_path: Path,
    metrics: pd.DataFrame,
    segment: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    """외부 자원 없는 단일 HTML 대시보드를 저장한다."""
    m = metrics.set_index("metric")["value"]
    normal = segment[segment["view"] == "normal"].copy()
    outlier = segment[segment["view"] == "outlier"].copy()
    flagged = normal[normal["stage"] != "normal"].copy()
    flagged["경과(분)"] = (flagged["relative_sec"] / 60).round(2)
    flagged["Ridge"] = flagged["ridge_max"].round(2)
    flagged["MCD"] = flagged["mcd_max"].round(2)
    flagged["AND"] = flagged["alarm_max"].round(2)
    flagged["단계"] = flagged["stage"].map({"caution": "주의", "alarm": "경보"})
    flagged = flagged[["seg_uid", "경과(분)", "state_name", "Ridge", "MCD", "AND", "단계"]]
    flagged.columns = ["세그먼트", "경과(분)", "운전 상태", "Ridge max", "MCD max", "AND max", "판정"]

    anomaly = outlier.sort_values("relative_sec").copy()
    anomaly["경과(초)"] = anomaly["relative_sec"].round(1)
    anomaly["Ridge"] = anomaly["ridge_max"].round(1)
    anomaly["MCD"] = anomaly["mcd_max"].round(1)
    anomaly["AND"] = anomaly["alarm_max"].round(1)
    anomaly = anomaly[["seg_uid", "경과(초)", "n_windows", "Ridge", "MCD", "AND"]]
    anomaly.columns = ["세그먼트", "경과(초)", "창 수", "Ridge max", "MCD max", "AND max"]

    cards = [
        ("주의 FPR", f"{100*m['주의 세그먼트 FPR']:.2f}%", "Ridge 단독 · time-block"),
        ("경보 FPR", f"{100*m['경보 세그먼트 FPR']:.2f}%", "Ridge ∩ MCD · time-block"),
        ("실제 이상", f"{int(m['경보 이상 탐지'])}/17", "판정 가능한 세그먼트 · 단일 이벤트"),
        ("판정 지연", f"{m['경보 지연 중앙값']:.1f}s", "1초 창 · burst 내부 중앙값"),
    ]
    card_html = "".join(
        f"<article class='metric'><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong>"
        f"<small>{html.escape(note)}</small></article>" for title, value, note in cards
    )
    check_view = checks.copy()
    check_view["passed"] = check_view["passed"].map({True: "통과", False: "실패"})
    check_view.columns = ["검증", "결과", "상세"]

    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ridge·MCD 2단계 경보 대시보드</title>
<style>
:root{{--ink:#172033;--muted:#687386;--line:#dce2ea;--paper:#f4f7fb;--blue:#2f6fed;--teal:#00a896;--amber:#f5a623;--red:#d7263d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif}}
.wrap{{max-width:1180px;margin:auto;padding:28px 22px 52px}} header{{background:linear-gradient(135deg,#102447,#1f4b83);color:white;border-radius:18px;padding:28px 30px;box-shadow:0 12px 30px #18335a22}}
.eyebrow{{font-size:12px;letter-spacing:.11em;text-transform:uppercase;color:#bcd3ff}} h1{{font-size:30px;margin:7px 0}} header p{{margin:0;color:#dbe8ff;line-height:1.65}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}} .badge{{background:#ffffff18;border:1px solid #ffffff2d;padding:6px 10px;border-radius:999px;font-size:12px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}} .metric{{background:white;border:1px solid var(--line);border-radius:14px;padding:17px;box-shadow:0 5px 18px #2538580b}}
.metric span,.metric small{{display:block;color:var(--muted);font-size:12px}} .metric strong{{display:block;font-size:28px;margin:5px 0}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}} button{{border:1px solid var(--line);background:white;color:var(--ink);padding:10px 15px;border-radius:10px;cursor:pointer;font-weight:700}} button.active{{background:var(--blue);color:white;border-color:var(--blue)}}
.panel{{display:none}} .panel.active{{display:block}} section{{background:white;border:1px solid var(--line);border-radius:16px;padding:22px;margin:14px 0;box-shadow:0 5px 18px #2538580b}}
h2{{font-size:20px;margin:0 0 8px}} h3{{font-size:16px;margin-top:24px}} .sub{{color:var(--muted);font-size:13px;line-height:1.6}} img{{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;margin-top:10px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;align-items:stretch;margin:18px 0}} .step{{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fafcff;font-size:13px}} .step b{{display:block;margin-bottom:5px}} .step.caution{{border-top:4px solid var(--amber)}} .step.alarm{{border-top:4px solid var(--red)}}
.data{{border-collapse:collapse;width:100%;font-size:12px;margin-top:12px}} .data th,.data td{{border-bottom:1px solid var(--line);padding:8px;text-align:right}} .data th:first-child,.data td:first-child{{text-align:left}} .data th{{background:#f7f9fc}}
.callout{{border-left:4px solid var(--amber);background:#fff8e8;padding:12px 15px;border-radius:8px;line-height:1.65;font-size:13px}} .limit{{border-left-color:var(--red);background:#fff1f3}}
code{{background:#edf2f8;padding:2px 5px;border-radius:5px}} footer{{color:var(--muted);font-size:12px;line-height:1.7;margin-top:20px}}
@media(max-width:780px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.flow{{grid-template-columns:1fr}}h1{{font-size:24px}}}}
</style></head><body><main class="wrap">
<header><div class="eyebrow">Static replay · Final operating rule</div><h1>Ridge 주의 → MCD 확인 2단계 경보</h1>
<p>저장된 23번 교차검증 점수로 정상 운전과 실제 이상 이벤트를 재생한다. 판정은 학습 정상 q99로 정한 <b>z &gt; 1</b>을 그대로 사용한다.</p>
<div class="badges"><span class="badge">1초 창</span><span class="badge">burst 경계 초기화</span><span class="badge">외부 서버·계정 없음</span><span class="badge">단일 HTML</span></div></header>
<div class="metrics">{card_html}</div>
<nav><button class="active" data-panel="overview">운영 개요</button><button data-panel="normal">정상 스트림</button><button data-panel="outlier">이상 이벤트</button><button data-panel="cases">오경보 사례</button><button data-panel="deploy">현장 확장</button></nav>
<div id="overview" class="panel active"><section><h2>최종 운영점</h2><p class="sub">Ridge는 민감한 후보를 먼저 올리고 MCD는 다른 통계 관점에서 같은 창을 확인한다. 경보는 주의의 부분집합이다.</p>
<div class="flow"><div class="step"><b>① 센서 입력</b>진동 2채널 + 전류 1채널, 10 Hz</div><div class="step"><b>② causal 변환</b>burst별 1차 차분·경계 초기화</div><div class="step caution"><b>③ 주의</b>Ridge 예측오차 z &gt; 1</div><div class="step alarm"><b>④ 경보</b>같은 창에서 Ridge와 MCD 모두 z &gt; 1</div><div class="step"><b>⑤ 대응 [가정]</b>주의는 추세 확인, 경보는 즉시 설비 확인</div></div>
<img src="{_image_uri(fig_dir / 'dashboard_preview.png')}" alt="최종 운영점 요약 그림">
<div class="callout">경보 FPR 0.25%는 time-block 평가의 세그먼트 평균이다. 이상 17세그먼트는 독립 고장 17건이 아니라 <b>하나의 실제 이상 이벤트를 나눈 조각</b>이다.</div></section></div>
<div id="normal" class="panel"><section><h2>정상 운전 스트림 재생</h2><p class="sub">time-block 평가 블록 1~4의 정상 417세그먼트, seed 0. 색은 정상·주의·경보를 나타낸다. 저부하에서는 최종 경보가 없었다.</p>
<img src="{_image_uri(fig_dir / 'normal_replay.png')}" alt="정상 운전 점수 및 상태 타임라인"></section></div>
<div id="outlier" class="panel"><section><h2>실제 이상 이벤트 재생</h2><p class="sub">같은 이상 이벤트가 각 fold에서 반복 평가되므로, 가장 많은 과거 정상으로 학습한 forward fold 4·seed 0 한 벌만 표시한다. 17개 판정 가능 세그먼트가 모두 경보에 포함됐다.</p>
<img src="{_image_uri(fig_dir / 'outlier_replay.png')}" alt="실제 이상 이벤트 창별 점수"><h3>세그먼트별 최대 점수</h3>{_table(anomaly)}</section></div>
<div id="cases" class="panel"><section><h2>정상에서 발생한 주의·경보</h2><p class="sub">Ridge와 MCD의 세그먼트 최대가 모두 1을 넘더라도 같은 창에서 넘지 않으면 경보가 아니다. <code>normal_331</code>이 그 예다. 최종 경보 오경보는 <code>normal_178</code> 한 건이다.</p>{_table(flagged)}
<h3>표시 계약 자동 검증</h3>{_table(check_view)}</section></div>
<div id="deploy" class="panel"><section><h2>운영 배포 구조 [가정]</h2><p class="sub">이번 제출물에서 구현·검증한 범위는 점수 CSV를 재생하는 정적 대시보드까지다. 아래 저장·모니터링·알림 계층은 현장 배포 시의 확장안이다.</p>
<div class="flow"><div class="step"><b>센서·DAQ</b>10 Hz 진동·전류</div><div class="step"><b>판정 서비스</b>Ridge 주의 + MCD 확인</div><div class="step"><b>점수·이벤트 저장</b>시계열 DB 또는 현장 저장소</div><div class="step"><b>모니터링 패널</b>정적 HTML과 같은 점수·상태·경보 표시</div><div class="step"><b>알림·조치</b>현장 확인 및 정비 기록 연결</div></div>
<div class="callout">Grafana 같은 도구는 가능한 모니터링 인터페이스의 예일 뿐이며 본 제출물에서 구축하거나 성능을 검증하지 않았다.</div></section></div>
<footer><div class="callout limit"><b>해석 한계.</b> 실제 설비 통신·시계열 DB·알림 전송을 연결한 실시간 시스템이 아니다. 정상 하루와 이상 이벤트 한 건의 저장 데이터를 순서대로 재생했다. 모델은 고장 부품·남은 수명을 진단하지 않는다. 센서 부착 위치가 확정되지 않아 탐지 대상을 유압펌프 모터-펌프 구동부 이상으로 한정한다.</div>
<p>재현: <code>notebooks/37_ridge_mcd_dashboard_JSC.ipynb</code> 실행 → <code>figures/37_ridge_mcd_dashboard_JSC/dashboard.html</code>. 외부 웹 자원·상용 라이선스·별도 대시보드 서버가 필요하지 않다.</p></footer>
</main><script>
document.querySelectorAll('nav button').forEach(btn=>btn.addEventListener('click',()=>{{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));btn.classList.add('active');document.getElementById(btn.dataset.panel).classList.add('active');}}));
</script></body></html>"""
    out_path.write_text(page, encoding="utf-8")


def build_dashboard(fig_dir: Path, res_dir: Path) -> dict[str, object]:
    """대시보드 산출물 전체를 만들고 노트북 표시용 객체를 반환한다."""
    data = load_inputs()
    replay, segment = prepare_replay(data["scores"], data["segments"])
    metrics = dashboard_metrics(data)
    checks = validate_contract(replay, segment, metrics)

    plot_normal_replay(segment, fig_dir)
    plot_outlier_replay(replay, fig_dir)
    plot_preview(metrics, fig_dir)
    metrics.to_csv(res_dir / "dashboard_summary.csv", index=False, encoding="utf-8-sig")
    segment.to_csv(res_dir / "replay_segments.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(res_dir / "contract_checks.csv", index=False, encoding="utf-8-sig")
    out_path = fig_dir / "dashboard.html"
    write_dashboard(fig_dir, out_path, metrics, segment, checks)
    return {"metrics": metrics, "replay": replay, "segments": segment, "checks": checks, "html": out_path}
