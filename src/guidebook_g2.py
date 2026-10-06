"""가이드북 LSTM-Autoencoder에 팀 계약을 적용한 G2 실행·평가.

`benchmark_jiw`와 같은 분할(group_kfold_seg 5 + time_block 4)·임계(train 정상 q99)·지표(evaluate)를 쓰고,
윈도우 목록은 11 parquet의 (`seg_uid`, `t_start`, `win_s`)를 그대로 따른다. 비교표에 다른 모델과 같은 행 형식으로 들어간다.

가이드북과 달라지는 점
- 입력: `preprocess.preprocess()` 결과(형식 통일 + 세그먼트 평균 제거), abs를 쓰지 않는다.
- 정규화: 채널별 MinMax를 train 정상 세그먼트에서만 fit (가이드북 코드 20과 같은 원칙).
- 시퀀스: 세그먼트 안에서만 20개(2초) — 가이드북은 burst 공백을 넘어 행을 이어 붙였다.
  평가 윈도우(1·2초)는 11 parquet과 동일한 세그먼트 안 윈도우이고, 점수는 윈도우 안 시점별 재구성 오차의 평균이다.
- 임계: train 정상 점수 q99. 이상 데이터를 임계 결정에 쓰지 않는다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

import benchmark_jiw as B
import evaluate as ev
import guidebook_lstm_ae as G
import paths

FS = 10.0


class Scaler:
    """채널별 MinMax (train 정상 세그먼트로만 fit). 범위를 벗어난 값은 그대로 두어 이상 크기가 잘리지 않게 한다."""

    def fit(self, segs):
        x = np.concatenate(segs)
        self.lo, self.hi = x.min(0), x.max(0)
        return self

    def __call__(self, x):
        return ((x - self.lo) / (self.hi - self.lo + 1e-9)).astype(np.float32)


def train_sequences(segs, seq=G.SEQ, stride=1):
    """세그먼트 안에서만 길이 seq 시퀀스를 만든다 (공백을 넘지 않음)"""
    out = []
    for x in segs:
        for s in range(0, len(x) - seq + 1, stride):
            out.append(x[s:s + seq])
    return np.stack(out).astype(np.float32)


def window_scores(net, scaler, segs, wins, mode="mean"):
    """윈도우(seg_uid, t_start, win_s)별 점수: 윈도우 안 시점별 재구성 오차의 평균.
    LSTM-AE 입력은 윈도우 안에서 seq(=20) 길이 시퀀스를 1칸씩 밀어 만들고(윈도우가 20보다 짧으면 앞뒤를 반복 패딩), 각 시퀀스의 재구성 오차를 평균한다."""
    w = int(round(wins["win_s"].iloc[0] * FS))
    starts = np.round(wins["t_start"].to_numpy() * FS).astype(int)
    cache = {}
    for u in wins["seg_uid"].unique():
        x = scaler(segs[u])
        seq = G.SEQ
        if len(x) < seq:                                   # 세그먼트가 시퀀스보다 짧으면 마지막 값을 반복해 채운다
            x = np.concatenate([x, np.repeat(x[-1:], seq - len(x), 0)])
        X = np.stack([x[i:i + seq] for i in range(len(x) - seq + 1)]).astype(np.float32)
        # 시점별 오차: 각 시퀀스 재구성 오차를 시점에 배정 (마지막 시점 기준, 가이드북과 같은 정의)
        e = G.recon_error(net, X, mode="last")
        pt = np.full(len(segs[u]), np.nan)
        pt[seq - 1:seq - 1 + len(e)] = e[:len(pt) - (seq - 1)] if len(pt) >= seq else e[:1]
        cache[u] = pt
    out = []
    for u, s in zip(wins["seg_uid"], starts):
        v = cache[u][s:s + w]
        v = v[~np.isnan(v)]
        out.append(v.mean() if len(v) else np.nan)
    sc = np.array(out)
    if np.isnan(sc).any():                                  # 윈도우 안에 시퀀스 끝점이 없으면 가장 가까운 시점 값으로 채운다
        for i in np.where(np.isnan(sc))[0]:
            u, s = wins["seg_uid"].iloc[i], starts[i]
            v = cache[u]
            valid = np.where(~np.isnan(v))[0]
            sc[i] = v[valid[np.argmin(np.abs(valid - (s + w - 1)))]] if len(valid) else 0.0
    return sc


G2_EPOCHS, G2_PATIENCE_ES, G2_PATIENCE_LR = 100, 20, 10     # 가이드북은 800 / 120 / 50 (분할 18회 × 시드를 현실적으로 돌리기 위해 축소)


def run_g2(nb, win_list=(1.0, 2.0), seeds=(0,), epochs=G2_EPOCHS, retrain=False, progress=True, segs=None, W=None,
           val_frac=0.1, patience_es=G2_PATIENCE_ES, patience_lr=G2_PATIENCE_LR):
    """G2: 모델 LSTM-AE × 윈도우 × 분할 9 × 시드. 반환 (cv_scores, error_cases, scores).
    학습 설정은 가이드북(최대 800 epoch, 조기종료 patience 120, 학습률 감소 patience 50)을 축소했다(100 / 20 / 10).
    train 시퀀스가 세그먼트 안에서만 만들어져 가이드북보다 적고, 분할이 18회라 원 설정으로는 하루 이상 걸리기 때문이다."""
    if segs is None or W is None:
        segs, W = B.load_inputs()
    attr = B._seg_attr(W)
    jobs = []
    for win_s in win_list:
        Wv = W[W["win_s"] == win_s].reset_index(drop=True)
        for sp, k, tr, te in B.iter_splits(Wv):
            for seed in seeds:
                jobs.append((win_s, sp, k, tr, te, seed))
    rows, errs, scores = [], [], []
    bar = tqdm(jobs, desc=nb, dynamic_ncols=True, disable=not progress)
    for win_s, sp, k, tr, te, seed in bar:
        model_id = f"lstm_ae_guidebook_w{win_s:g}s"
        bar.set_postfix_str(f"{model_id} {sp} f{k} s{seed}")
        path = paths.MODELS / nb / f"{model_id}_{sp}_f{k}_s{seed}.pt"
        tr_uids = list(tr["seg_uid"].unique())
        t0 = time.time()
        if path.exists() and not retrain:
            st = torch.load(path, weights_only=False)
            net = G.LSTMAE(); net.load_state_dict(st["state"]); net.eval()
            scaler = Scaler(); scaler.lo, scaler.hi = st["lo"], st["hi"]
            status = "loaded"
        else:
            scaler = Scaler().fit([segs[u] for u in tr_uids])
            rng = np.random.default_rng(seed)
            val_set = set(rng.choice(tr_uids, max(1, int(len(tr_uids) * val_frac)), replace=False))   # 조기종료용 검증(정상만)
            Xt = train_sequences([scaler(segs[u]) for u in tr_uids if u not in val_set])
            Xv = train_sequences([scaler(segs[u]) for u in tr_uids if u in val_set])
            net, hist = G.fit_lstm_ae(Xt, Xv, seed=seed, epochs=epochs, patience_es=patience_es, patience_lr=patience_lr,
                                      progress=progress, desc=model_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": net.state_dict(), "lo": scaler.lo, "hi": scaler.hi, "epochs_run": len(hist),
                        "n_train_seq": len(Xt)}, path)
            status = "trained"
        s_tr, s_te = window_scores(net, scaler, segs, tr), window_scores(net, scaler, segs, te)
        thr = ev.threshold_from_normal(s_tr)
        y = te["label"].to_numpy()
        n_te, o_te = te[y == 0], te[y == 1]
        sn, so = s_te[y == 0], s_te[y == 1]
        dly = ev.detection_delay_s(so, o_te["seg_uid"], o_te["t_start"], thr, win_s=win_s)
        det = dly["detected"].to_numpy(bool)
        st_ = n_te["state"].to_numpy()
        ep_run = int(torch.load(path, weights_only=False).get("epochs_run", -1)) if path.exists() else -1
        row = {"family": "Guidebook", "model": "LSTM-AE (가이드북, 팀 계약)", "key": "lstm_ae_g2", "feature_set": "",
               "epochs_run": ep_run,
               "model_id": model_id, "win_s": win_s, "split": sp, "fold": k, "seed": seed, "status": status,
               "auc": ev.auc(y, s_te), "fpr_sample": ev.fpr_sample(sn, thr),
               "fpr_segment": ev.fpr_segment(sn, n_te["seg_uid"], thr),
               "delay_median_s": float(np.nanmedian(dly["delay_in_window_s"])) if det.any() else np.nan,
               "felt_delay_median_s": float(np.nanmedian(dly["delay_in_window_s"])) + ev.BURST_GAP_S if det.any() else np.nan,
               "thr": thr, "n_out_seg": len(dly), "n_detected": int(det.sum()), "recall_window": float(np.mean(so > thr)),
               "fpr_high_load": float(np.mean(sn[st_ == 0] > thr)) if (st_ == 0).any() else np.nan,
               "fpr_low_load": float(np.mean(sn[st_ == 1] > thr)) if (st_ == 1).any() else np.nan}
        rows.append(row)
        if seed == seeds[0]:
            ex = pd.DataFrame({"seg_uid": n_te["seg_uid"].to_numpy(), "exceed": sn > thr})
            for u, v in ex.groupby("seg_uid")["exceed"].agg(["sum", "size"]).query("sum > 0").iterrows():
                errs.append({"model_id": model_id, "split": sp, "fold": k, "type": "FP", "seg_uid": u,
                             "n_exceed": int(v["sum"]), "n_windows": int(v["size"]),
                             "state": B.STATE_NAME.get(attr.loc[u, "state"], "")})
            for u in dly.loc[~det, "seg_uid"]:
                errs.append({"model_id": model_id, "split": sp, "fold": k, "type": "FN", "seg_uid": u,
                             "vib_grade": attr.loc[u, "vib_grade"], "cur_grade": attr.loc[u, "cur_grade"]})
            scores.append(pd.DataFrame({"model_id": model_id, "split": sp, "fold": k, "seg_uid": te["seg_uid"].to_numpy(),
                                        "label": y, "state": te["state"].to_numpy(), "score": s_te, "thr": thr}))
    return pd.DataFrame(rows), pd.DataFrame(errs), pd.concat(scores, ignore_index=True)
