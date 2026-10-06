"""가이드북 LSTM-Autoencoder 원본 방식(G1) 실행과 평가.

가이드북 27~38쪽의 절차를 그대로 따른다(자세한 대응은 `guidebook_lstm_ae.py` 머리말).
- 학습 정상 앞 15,000행, 평가 정상 뒤 5,000행 + 이상 전체, 검증 = 평가 정상 앞 880 + 이상 앞 300 시퀀스
- 임계 = 검증 세트에서 정밀도 = 재현율이 되는 점수 → **이상 데이터를 임계 결정에 쓴다**(팀 계약 위반, 비교용 재현)
- 학습: Adam 1e-3, MSE, 배치 128, 최대 800 epoch, EarlyStopping(120), ReduceLROnPlateau(0.7, 50)
실행: `python src/guidebook_g1.py <seed>` — 시드 하나를 학습하고 가중치와 결과 한 줄을 저장한다(CPU 약 85~150분).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn import metrics as skm

import data_quality as dq
import guidebook_lstm_ae as G
import paths

NB = "29_model_guidebook_lstm_ae_JIW"
GUIDEBOOK_REPORTED = {"TN": 3922, "FP": 78, "FN": 26, "TP": 154}   # 가이드북 38~39쪽 보고 결과


def _data():
    d = dq.load_all()
    return G.guidebook_sequences(d["normal"], d["outlier"])


def _summ(seed, net, hist_len, Xv, yv, Xs, ys) -> dict:
    ev, es = G.recon_error(net, Xv), G.recon_error(net, Xs)
    thr, p, r = G.pr_equal_threshold(yv, ev)
    pred = (es > thr).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(ys, pred).ravel()
    return {"seed": seed, "epochs_run": hist_len, "threshold": thr, "valid_precision": p, "valid_recall": r,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "accuracy": skm.accuracy_score(ys, pred),
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn), "f1": skm.f1_score(ys, pred),
            "auc": skm.roc_auc_score(ys, es), "fpr": fp / (fp + tn)}


def run_g1(seed: int = 0, epochs: int = 800, threads: int = 6) -> dict:
    """학습하고 가중치(`models/29_*/g1_s<seed>.pt`)와 결과(`results/29_*/g1_runs.csv`)를 저장한다."""
    torch.set_num_threads(threads)
    Xt, (Xv, yv), (Xs, ys) = _data()
    t0 = time.time()
    net, hist = G.fit_lstm_ae(Xt, Xv[yv == 0], seed=seed, epochs=epochs, progress=True, desc=f"G1 s{seed}")
    row = _summ(seed, net, len(hist), Xv, yv, Xs, ys)
    row["sec"] = round(time.time() - t0)
    mdir = paths.MODELS / NB
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "epochs_run": len(hist), "hist": hist}, mdir / f"g1_s{seed}.pt")
    out = paths.RESULTS / NB / "g1_runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([old[old["seed"] != seed] if len(old) else old, pd.DataFrame([row])], ignore_index=True).sort_values("seed").to_csv(out, index=False)
    return row


def eval_g1(seed: int) -> tuple[dict, np.ndarray]:
    """저장된 가중치로 같은 평가를 다시 한다(학습 없음). 반환: (결과 dict, 학습 곡선 [train, valid] 손실)"""
    st = torch.load(paths.MODELS / NB / f"g1_s{seed}.pt", weights_only=False)
    net = G.LSTMAE()
    net.load_state_dict(st["state"])
    net.eval()
    _, (Xv, yv), (Xs, ys) = _data()
    return _summ(seed, net, st["epochs_run"], Xv, yv, Xs, ys), st["hist"]


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(run_g1(s, threads=int(sys.argv[2]) if len(sys.argv) > 2 else 6), flush=True)
