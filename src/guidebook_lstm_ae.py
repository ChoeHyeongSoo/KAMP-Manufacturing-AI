"""주제 ③ — 공식 가이드북(2.3 분석 체험) LSTM-Autoencoder 재현과 팀 계약 적용 버전.

가이드북 원 방식(G1)과, 같은 모델에 우리 팀 계약을 적용한 방식(G2)을 나란히 돌려 "누수를 막으면 성능이 어떻게 변하나"를 보여준다.
모델 구조·학습 설정은 두 버전이 같고, 바뀌는 것은 입력 처리·시퀀스 구성·분할·임계뿐이다.

가이드북 원 방식 (G1), 가이드북 27~38쪽 코드 14~30
- 입력: 원시 3채널에 abs() (진폭 변환, 코드 14) → MinMaxScaler (train 정상으로만 fit, 코드 20)
- 분할: 정상 앞 15,000행 학습, 뒤 5,000행 + 이상 전체 평가 (코드 19). 세그먼트 구조를 보지 않고 행을 이어 붙인다.
- 시퀀스: 과거 20개(2초), 정답은 index+20+100 (코드 21, 설명 "100시점(10초) 이후의 이상작동"). LSTM-AE 학습·점수에는 정답을 쓰지 않는다.
- 검증: 평가 정상 앞 880, 이상 앞 300 시퀀스로 임계값을 정한다 → **이상 데이터를 임계 결정에 쓴다** (코드 22).
- 임계: 검증 세트의 정밀도 = 재현율이 되는 지점 (코드 29).
- 점수: 시퀀스 마지막 시점의 재구성 오차 (코드 28의 flatten).
- 학습: Adam 1e-3, MSE, batch 128, 최대 800 epoch, ReduceLROnPlateau(0.7, patience 50), EarlyStopping(patience 120, min_delta 1e-5).
  검증 손실은 검증 세트의 정상 시퀀스로 잰다 (코드 24, 26).

팀 계약 적용 (G2)
- 입력: `preprocess.preprocess()` (형식 통일 + 세그먼트 평균 제거). abs를 쓰지 않는다. 채널별 MinMax는 train 정상으로만 fit.
- 시퀀스: 세그먼트 안에서만 20개(2초) 윈도우. 점수는 시퀀스 마지막 시점의 재구성 오차(가이드북과 동일)와 시퀀스 평균 재구성 오차 두 가지를 낸다.
- 분할·임계·지표: `benchmark_jiw`와 같다(group_kfold_seg 5 + time_block 4, train 정상 q99, evaluate 지표).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics as skm
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

SEQ = 20          # 가이드북 sequence (2초)
OFFSET = 100      # 가이드북 정답 오프셋 (10초)
FS = 10.0


# ------------------------------------------------------------------ model
class LSTMAE(nn.Module):
    """가이드북 코드 25: 인코더 LSTM 64 → 32, RepeatVector, 디코더 LSTM 32 → 64, TimeDistributed Dense"""

    def __init__(self, n_features=3, seq=SEQ):
        super().__init__()
        self.seq = seq
        self.enc1 = nn.LSTM(n_features, 64, batch_first=True)
        self.enc2 = nn.LSTM(64, 32, batch_first=True)
        self.dec1 = nn.LSTM(32, 32, batch_first=True)
        self.dec2 = nn.LSTM(32, 64, batch_first=True)
        self.out = nn.Linear(64, n_features)

    def forward(self, x):                                   # x: (B, seq, F)
        h, _ = self.enc1(x)
        _, (h2, _) = self.enc2(h)                           # 마지막 은닉 상태 (return_sequences=False)
        z = h2[-1].unsqueeze(1).repeat(1, self.seq, 1)      # RepeatVector
        y, _ = self.dec1(z)
        y, _ = self.dec2(y)
        return self.out(y)


def fit_lstm_ae(X_train, X_val_normal, seed=0, epochs=800, batch=128, lr=1e-3, patience_es=120, patience_lr=50,
                min_delta=1e-5, progress=True, desc="LSTM-AE"):
    """가이드북 코드 26의 학습 설정 (Adam, MSE, ReduceLROnPlateau(0.7, 50), EarlyStopping(120, 1e-5, best 복원))"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = LSTMAE(X_train.shape[2], X_train.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=patience_lr)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val_normal, dtype=torch.float32)
    best, best_state, wait, hist = np.inf, None, 0, []
    bar = tqdm(range(epochs), desc=desc, leave=False, disable=not progress, dynamic_ncols=True)
    for ep in bar:
        net.train()
        perm, tot = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((net(Xt[idx]) - Xt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            vl = ((net(Xv) - Xv) ** 2).mean().item()
        sched.step(vl)
        hist.append((tot / len(Xt), vl))
        if vl < best - min_delta:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        bar.set_postfix(train=f"{hist[-1][0]:.5f}", val=f"{vl:.5f}", best=f"{best:.5f}")
        if wait >= patience_es:
            break
    net.load_state_dict(best_state)
    return net.eval(), np.array(hist)


def recon_error(net, X, mode="last", batch=2048):
    """시퀀스별 재구성 오차. mode="last": 마지막 시점(가이드북 flatten), "mean": 시퀀스 전체 평균"""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.tensor(X[i:i + batch], dtype=torch.float32)
            e = (net(x) - x) ** 2
            out.append((e[:, -1, :].mean(1) if mode == "last" else e.mean((1, 2))).numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ G1: 가이드북 원본
def guidebook_sequences(df_normal, df_outlier):
    """가이드북 코드 14~22: abs → MinMax(train fit) → 20개 시퀀스, 정답은 index+20+100.
    반환: X_train, (X_valid, y_valid), (X_test, y_test)"""
    use = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
    N, O = df_normal[use].abs(), df_outlier[use].abs()
    Xtr, Xte_n = N.iloc[:15000], N.iloc[15000:]
    sc = MinMaxScaler().fit(Xtr)

    def seq(X, label):
        X = sc.transform(X)
        n = len(X) - SEQ - OFFSET
        return np.stack([X[i:i + SEQ] for i in range(n)]).astype(np.float32), np.full(n, label)

    Xt, _ = seq(Xtr, 0)
    Xn, yn = seq(Xte_n, 0)
    Xa, ya = seq(O, 1)
    Xv = np.vstack([Xn[:880], Xa[:300]])
    yv = np.r_[yn[:880], ya[:300]]
    Xs = np.vstack([Xn[880:], Xa[300:]])
    ys = np.r_[yn[880:], ya[300:]]
    return Xt, (Xv, yv), (Xs, ys)


def pr_equal_threshold(y, score):
    """가이드북 코드 29: 정밀도 == 재현율이 되는 첫 지점. 정확히 같은 지점이 없으면 |p − r|이 가장 작은 지점"""
    p, r, t = skm.precision_recall_curve(y, score)
    hit = [i for i, (a, b) in enumerate(zip(p, r)) if a == b]
    i = hit[0] if hit else int(np.argmin(np.abs(p[:-1] - r[:-1])))
    return float(t[min(i, len(t) - 1)]), float(p[i]), float(r[i])
