"""주제 ③ — JIW 32 경보 근거 카드: 경보가 왜 났는지 코드로 뽑은 근거 + 템플릿 문장 + 점검 권고.

원칙(docs/final_design_JIW.md §2.5, §5)
- 근거 추출은 결정적 코드다. 어느 모델이 임계를 몇 배 넘었는지, CNN 예측 오차가 어느 센서에 몰렸는지,
  MCD 입력 피처 중 정상에서 가장 벗어난 것이 무엇인지만 말한다. 원인을 단정하지 않는다.
- 센서 부착 부위를 확정할 수 없으므로 "상부/하부 진동", "전류" 같은 **신호 이름**으로 표현한다. 모터인지 펌프인지는 말하지 않는다.
- 점검 권고는 가이드북·일반 설비 상식에서 온 **가정**이다(이 데이터로 검증하지 않았다). 문장에 "가정"을 붙인다.
- 문장은 템플릿이 기본이다. LLM으로 다듬는다면 이 카드의 숫자만 입력으로 줘야 한다(여기서는 구현하지 않음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import models_jiw as M
import preprocess

CH = preprocess.SENSORS                                   # AI0_Vibration, AI1_Vibration, AI2_Current
CH_KO = {CH[0]: "상부 진동", CH[1]: "하부 진동", CH[2]: "전류"}
CHECK = {CH[0]: "상단 체결부·커플링·정렬", CH[1]: "하단 베이스·마운트", CH[2]: "전원·부하 변동, 토출 압력",
         "all": "구동부 전반(즉시 정지 후 종합 점검)"}
EXPECTED = {"spike": {CH[0]}, "amplitude": {CH[0], CH[1]}, "antiphase": {CH[1]}, "current": {CH[2]}}   # benchmark_jiw.inject가 건드리는 채널
FEATURE_KO = {"rms": "RMS", "peak": "피크", "p2p": "피크-피크", "kurt": "첨도", "crest": "크레스트 팩터", "skew": "왜도"}


def cnn_step_err(cnn: M.CNNDeepAnT, x_raw: np.ndarray) -> np.ndarray:
    """시점별 채널별 정규화 제곱 오차 (n, 3). 앞 lookback개 시점은 nan. 합의 평균이 CNN 점수의 시점 점수와 같다."""
    x = cnn._norm(x_raw)
    L = cnn.lookback
    out = np.full((len(x), 3), np.nan)
    if len(x) <= L:
        return out
    import torch
    with torch.no_grad():
        X = torch.tensor(np.stack([x[t - L:t] for t in range(L, len(x))]))
        p, _ = cnn.net(X)
    out[L:] = ((p.numpy() - x[L:]) / cnn.stats["sd"]) ** 2
    return out


def window_channel_share(err: np.ndarray, t_start: float, win_s: float = 2.0, fs: float = 10.0) -> np.ndarray:
    """창 안 시점들의 채널별 오차 합을 1로 맞춘 비중 (3,)"""
    s, w = int(round(t_start * fs)), int(round(win_s * fs))
    e = np.nanmean(err[s:s + w], 0)
    return e / e.sum() if np.isfinite(e).all() and e.sum() > 0 else np.full(3, np.nan)


def mcd_top_features(mcd: M.MCD, feat_row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    """MCD 입력 피처를 train 정상 평균·표준편차로 표준화했을 때 절댓값이 큰 상위 k개 [(피처, 부호 있는 z)]"""
    x = mcd.scaler.transform(mcd._X(feat_row.to_frame().T))[0]
    idx = np.argsort(-np.abs(x))[:k]
    return [(mcd.cols[i], float(x[i])) for i in idx]


def feature_channel(name: str) -> str:
    return next((c for c in CH if name.startswith(c)), "")


def feature_label(name: str) -> str:
    ch = feature_channel(name)
    stat = name[len(ch) + 1:] if ch else name
    return f"{CH_KO.get(ch, '')} {FEATURE_KO.get(stat, stat)}".strip()


def make_card(level: str, seg_uid: str, t_start: float, z_cnn: float, z_mcd: float, share: np.ndarray,
              top_feats: list[tuple[str, float]], state_est: int) -> dict:
    """근거 카드 dict. level: '주의' | '경보'"""
    order = np.argsort(-np.nan_to_num(share, nan=0))
    top = CH[order[0]]
    second = CH[order[1]]
    vib = share[0] + share[1]
    if share[2] >= 0.25 and vib >= 0.5:
        check_key, where = "all", "진동과 전류"
    elif share[order[0]] < 0.5 and share[order[1]] >= 0.3:
        check_key, where = top, f"{CH_KO[top]}·{CH_KO[second]}"
    else:
        check_key, where = top, CH_KO[top]
    feats = ", ".join(f"{feature_label(n)} {z:+.1f}σ" for n, z in top_feats)
    state = "고부하" if state_est == 0 else "저부하"
    text = (f"[{level}] {where} 쪽 이상 의심. CNN 점수가 임계의 {z_cnn:.1f}배, MCD 점수가 임계의 {z_mcd:.1f}배. "
            f"예측 오차 비중: {CH_KO[CH[0]]} {share[0]:.0%} · {CH_KO[CH[1]]} {share[1]:.0%} · {CH_KO[CH[2]]} {share[2]:.0%}. "
            f"정상에서 가장 벗어난 통계: {feats}. 운전 상태 추정: {state}. 점검 권고(가정): {CHECK[check_key]}.")
    return {"level": level, "seg_uid": seg_uid, "t_start": t_start, "z_cnn": z_cnn, "z_mcd": z_mcd,
            "share_ai0": float(share[0]), "share_ai1": float(share[1]), "share_ai2": float(share[2]),
            "top_channel": top, "where": where, "top_features": ";".join(f"{n}:{z:+.2f}" for n, z in top_feats),
            "state_est": state, "check": CHECK[check_key], "text": text}


def level_of(z_cnn: float, z_mcd: float) -> str:
    """28번의 포함 관계: 경보 = CNN AND MCD 초과, 주의 = CNN 단독 초과, 그 외 정상"""
    if z_cnn > 1 and z_mcd > 1:
        return "경보"
    return "주의" if z_cnn > 1 else "정상"


# ------------------------------------------------------------------ 평가: 합성 이상으로 근거 정합성, 실제 이상 카드
def _models(sp, k, seed, win_s=2.0):
    import ensemble_jiw as E
    return E._load("cnn", win_s, sp, k, seed), E._load("mcd", win_s, sp, k, seed)


def synthetic_attribution(segs, W, seed=0, sevs=(1.0, 2.0, 4.0, 8.0), win_s=2.0, progress=True) -> pd.DataFrame:
    """test 정상 윈도우에 합성 이상을 넣고, 카드가 나올 창(CNN z > 1)에서 근거가 주입한 채널을 가리키는지 센다.
    CNN 근거 = 예측 오차 비중 1위 채널, MCD 근거 = 표준화 편차 1위 피처의 채널. group_kfold 5개 fold, 임계는 train 정상 q99."""
    import benchmark_jiw as B
    import evaluate as ev
    from tqdm.auto import tqdm
    Wv = W[W["win_s"] == win_s].reset_index(drop=True)
    rows = []
    for sp, k, tr, te in tqdm(list(B.iter_splits(Wv)), desc="attribution", disable=not progress):
        if sp != "group_kfold_seg":
            continue
        cnn, mcd = _models(sp, k, seed, win_s)
        tc, tm = ev.threshold_from_normal(cnn.score(segs, tr)), ev.threshold_from_normal(mcd.score_features(tr))
        n_te = te[te["label"] == 0].reset_index(drop=True)
        sd = np.concatenate([segs[u] for u in tr["seg_uid"].unique()]).std(0)
        for kind in B.INJ_TYPES:
            for sev in sevs:
                rng = np.random.default_rng(100 + k)
                inj = dict(segs)
                for u in n_te["seg_uid"].unique():
                    inj[u] = B.inject(segs[u], kind, sev, sd, rng)
                F = B.recompute_features(inj, n_te, B.AMP_COLS)
                zc, zm = cnn.score(inj, n_te) / tc, mcd.score_features(F) / tm
                errs = {u: cnn_step_err(cnn, inj[u]) for u in n_te["seg_uid"].unique()}
                for i, r in n_te.iterrows():
                    if zc[i] <= 1:
                        continue
                    share = window_channel_share(errs[r["seg_uid"]], r["t_start"], win_s)
                    feats = mcd_top_features(mcd, F.iloc[i], 1)
                    exp = EXPECTED[kind]
                    rows.append({"fold": k, "kind": kind, "sev": sev, "level": level_of(zc[i], zm[i]),
                                 "cnn_top": CH[int(np.nanargmax(share))], "mcd_top": feature_channel(feats[0][0]),
                                 "cnn_hit": CH[int(np.nanargmax(share))] in exp, "mcd_hit": feature_channel(feats[0][0]) in exp})
    return pd.DataFrame(rows)


def real_cards(segs, W, seed=0, win_s=2.0, progress=True) -> pd.DataFrame:
    """실제 이상 창과 오경보(AND 경보가 난 정상 창)의 근거 카드. group_kfold 5개 fold, 임계는 train 정상 q99."""
    import benchmark_jiw as B
    import ensemble_jiw as E
    import evaluate as ev
    from tqdm.auto import tqdm
    Wv = W[W["win_s"] == win_s].reset_index(drop=True)
    cards = []
    for sp, k, tr, te in tqdm(list(B.iter_splits(Wv)), desc="cards", disable=not progress):
        if sp != "group_kfold_seg":
            continue
        cnn, mcd = _models(sp, k, seed, win_s)
        tc, tm = ev.threshold_from_normal(cnn.score(segs, tr)), ev.threshold_from_normal(mcd.score_features(tr))
        te = te.reset_index(drop=True)
        zc, zm = cnn.score(segs, te) / tc, mcd.score_features(te) / tm
        st = E.est_state(te)
        errs = {u: cnn_step_err(cnn, segs[u]) for u in te["seg_uid"].unique()}
        for i, r in te.iterrows():
            lv = level_of(zc[i], zm[i])
            if lv != "경보" and not (r["label"] == 1 and lv != "정상"):
                continue
            share = window_channel_share(errs[r["seg_uid"]], r["t_start"], win_s)
            c = make_card(lv, r["seg_uid"], float(r["t_start"]), float(zc[i]), float(zm[i]), share,
                          mcd_top_features(mcd, te.iloc[i][mcd.cols]), int(st[i]))
            c.update({"fold": k, "label": int(r["label"]), "vib_grade": r["vib_grade"], "cur_grade": r["cur_grade"],
                      "state": r["state"]})
            cards.append(c)
    return pd.DataFrame(cards)


def plot_attribution(attr: pd.DataFrame, fig_dir, name: str = "attribution.png"):
    """주입 유형별 근거 적중률: CNN 오차 비중 1위 채널 / MCD 표준화 편차 1위 피처의 채널이 주입 채널과 같은 비율"""
    import matplotlib.pyplot as plt
    import benchmark_jiw as B
    B._style()
    kinds = list(B.INJ_TYPES)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, kind in zip(axes, kinds):
        d = attr[attr["kind"] == kind].groupby("sev")[["cnn_hit", "mcd_hit"]].mean()
        ax.plot(d.index, d["cnn_hit"], marker="o", ms=4, color="#2a78d6", label="CNN 오차 비중")
        ax.plot(d.index, d["mcd_hit"], marker="s", ms=4, color="#eb6834", label="MCD 편차 피처")
        ax.set_xscale("log", base=2)
        ax.set_xticks(d.index, [f"{v:g}" for v in d.index])
        ax.set_title(B.INJ_KO[kind], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("합성 강도")
    axes[0].set_ylabel("주입 채널 적중률")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("근거 카드가 주입한 채널을 가리키는 비율 (CNN 경보가 난 창)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=120)
    plt.close(fig)
