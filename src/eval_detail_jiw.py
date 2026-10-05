"""주제 ③ — JIW 35 평가 상세: 혼동행렬(세그먼트·창), ROC·PR, 점수 분포, 임계 안정성, 모델 크기·학습량.

28번이 만든 점수 캐시(`data/processed/28_ensemble_scores_real.parquet`)만 읽는다. 새로 학습하지 않는다.
혼동행렬의 정의는 팀 평가 모듈(`evaluate.segment_confusion`)과 같다: 세그먼트는 2초 윈도우가 하나 이상 있는 것만 세고,
세그먼트 경보 = 그 세그먼트의 윈도우 중 하나라도 임계 초과. 양성 = 이상.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import benchmark_jiw as B
import ensemble_jiw as E

KEYS = E.KEYS
RULE_KO = {"cnn": "CNN 단독 (주의)", "mcd": "MCD 단독", "cnn&mcd": "CNN AND MCD (경보)", "cnn|mcd": "CNN OR MCD"}


def _metrics(tp, fp, fn, tn) -> dict:
    sd = lambda a, b: a / b if b else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "정밀도": sd(tp, tp + fp), "재현율": sd(tp, tp + fn), "특이도": sd(tn, tn + fp),
            "F1": sd(2 * tp, 2 * tp + fp + fn), "오경보율": sd(fp, fp + tn)}


def confusion(real: pd.DataFrame, rules, split: str, level: str = "segment") -> pd.DataFrame:
    """규칙별 혼동행렬(시드 평균 개수 → 지표). group_kfold: 5 fold 합산(모든 세그먼트가 한 번씩 test).
    time_block: 정상은 4 block 합산, 이상은 매 block 전체가 test에 들어가므로 block 평균(반복 계수 방지)."""
    rows = []
    d0 = real[real["split"] == split]
    for r in rules:
        d = d0.assign(_hit=E.rule_z(d0, r) > 1)
        if level == "segment":
            d = d.groupby(KEYS + ["seg_uid", "label"])["_hit"].any().reset_index()
        per_seed = []
        for seed, g in d.groupby("seed"):
            n, o = g[g["label"] == 0], g[g["label"] == 1]
            fp, tn = int(n["_hit"].sum()), int((~n["_hit"]).sum())
            if split == "group_kfold_seg":
                tp, fn = int(o["_hit"].sum()), int((~o["_hit"]).sum())
            else:
                nf = o["fold"].nunique()
                tp, fn = o["_hit"].sum() / nf, (~o["_hit"]).sum() / nf
            per_seed.append((tp, fp, fn, tn))
        tp, fp, fn, tn = np.mean(per_seed, 0)
        rows.append({"규칙": RULE_KO[r], "단위": "세그먼트" if level == "segment" else "윈도우", "분할": split, **_metrics(tp, fp, fn, tn)})
    return pd.DataFrame(rows)


def thresholds_by_fold(real: pd.DataFrame) -> pd.DataFrame:
    """fold·시드별 CNN·MCD 임계(train 정상 q99). 임계가 fold마다 얼마나 흔들리는지"""
    c = ["thr_g_cnn", "thr_g_mcd", "thr_g_lstm", "thr_g_gat"]
    return real.groupby(KEYS)[c].first().reset_index()


# ------------------------------------------------------------------ 그림
def plot_confusion_grid(conf: pd.DataFrame, fig_dir, title: str, name: str):
    import matplotlib.pyplot as plt
    B._style()
    n = len(conf)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    for ax, (_, r) in zip(np.atleast_1d(axes), conf.iterrows()):
        m = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(np.log1p(m), cmap="Blues", vmin=0, vmax=np.log1p(m.max()) * 1.15)
        for i in range(2):
            for j in range(2):
                v = m[i, j]
                ax.text(j, i, f"{v:.1f}" if abs(v - round(v)) > 1e-9 else f"{int(v)}", ha="center", va="center", fontsize=11,
                        color="white" if np.log1p(v) > np.log1p(m.max()) * 0.6 else "#222")
        ax.set_xticks([0, 1], ["정상 판정", "이상 판정"], fontsize=8)
        ax.set_yticks([0, 1], ["실제 정상", "실제 이상"], fontsize=8)
        ax.set_title(f"{r['규칙']}\n정밀도 {r['정밀도']:.2f} · 재현율 {r['재현율']:.2f} · F1 {r['F1']:.2f}", fontsize=8.5)
        ax.grid(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_roc_pr(real: pd.DataFrame, rules, fig_dir, split="group_kfold_seg", seed=0, name="roc_pr.png"):
    """창 단위 ROC(x축 로그)와 PR 곡선. 별 = 임계(z = 1)에서의 운영점"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
    B._style()
    plt.rcParams["axes.formatter.use_mathtext"] = True   # 로그 축 위첨자 마이너스 글리프 깨짐 방지
    g = real[(real["split"] == split) & (real["seed"] == seed)]
    y = g["label"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for r, col in zip(rules, ["#2a78d6", "#8a8984", "#eb6834", "#d6332a"]):
        z = E.rule_z(g, r)
        fpr, tpr, _ = roc_curve(y, z)
        a1.plot(np.maximum(fpr, 1e-4), tpr, color=col, lw=1.6, label=f"{RULE_KO[r]} (AUC {roc_auc_score(y, z):.4f})")
        hit = z > 1
        a1.plot(max(hit[y == 0].mean(), 1e-4), hit[y == 1].mean(), "*", color=col, ms=11)
        p, rc, _ = precision_recall_curve(y, z)
        a2.plot(rc, p, color=col, lw=1.6, label=RULE_KO[r])
        a2.plot(hit[y == 1].mean(), (hit & (y == 1)).sum() / max(hit.sum(), 1), "*", color=col, ms=11)
    a1.set_xscale("log")
    a1.set_xlabel("오경보율 (정상 창, 로그 축)")
    a1.set_ylabel("재현율 (이상 창)")
    a1.set_title("ROC (창 단위, 별 = 임계 운영점)", loc="left", fontsize=9.5)
    a1.legend(frameon=False, fontsize=7.5, loc="lower right")
    a2.set_xlabel("재현율")
    a2.set_ylabel("정밀도")
    a2.set_ylim(0, 1.05)
    a2.set_title("정밀도-재현율 (창 단위)", loc="left", fontsize=9.5)
    a2.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(f"{split}, 시드 {seed}, 5 fold 합침 (이상 창 수는 정상 창 대비 적어 정밀도는 낙관적일 수 있음)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def plot_score_hist(real: pd.DataFrame, fig_dir, split="group_kfold_seg", seed=0, name="score_hist.png"):
    """CNN·MCD의 임계 대비 배수(z) 분포: 정상 vs 실제 이상. 점선 = 임계(z = 1)"""
    import matplotlib.pyplot as plt
    B._style()
    plt.rcParams["axes.formatter.use_mathtext"] = True   # 로그 축 위첨자 마이너스 글리프 깨짐 방지
    g = real[(real["split"] == split) & (real["seed"] == seed)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, m, nm in zip(axes, ["cnn", "mcd"], ["CNN (예측 오차)", "MCD (통계 이탈)"]):
        z = E.z_scores(g, m)
        bins = np.logspace(-2.5, 2.7, 60)
        ax.hist(z[g["label"] == 0], bins=bins, color="#8a8984", alpha=0.75, label="정상 창")
        ax.hist(z[g["label"] == 1], bins=bins, color="#d6332a", alpha=0.75, label="실제 이상 창")
        ax.axvline(1, color=B.INK, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("점수 / 임계 (1 = 임계)")
        ax.set_ylabel("창 수")
        ax.set_title(nm, loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("정상과 실제 이상의 점수 분포 (임계를 크게 넘는 이상 창이 많다)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)


def model_sizes(lookback: int = 5, win: int = 20) -> pd.DataFrame:
    """네트워크별 학습 파라미터 수"""
    import models_jiw as M
    nets = {"CNN (DeepAnT 변형)": M.CNNNet(lookback), "LSTM-AD": M.LSTMNet(lookback), "MTAD-GAT": M.MTADGATNet(lookback),
            "GDN": M.GDNNet(lookback), "DeepSVDD": M.SVDDNet(win)}
    return pd.DataFrame({"모델": list(nets), "파라미터 수": [sum(p.numel() for p in n.parameters()) for n in nets.values()]})
