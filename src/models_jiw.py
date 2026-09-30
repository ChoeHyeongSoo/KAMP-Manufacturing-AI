"""주제 ③ — 정상 데이터 기반 이상탐지 모델 (24·25·26 노트북 공통).

계열 구분은 KDD'25 서베이(Paparrizos et al.)의 분류를 따른다.
- Forecasting : CNN(DeepAnT 구조), LSTM-AD — 과거 `lookback` 샘플로 다음 샘플을 예측, 오차 = 점수
- Graph       : MTAD-GAT(센서·시점 graph attention + GRU, 예측+복원), GDN(센서 임베딩 그래프, 센서별 편차 최댓값)
- Distribution: DeepSVDD(원신호 윈도우) / MCD·OCSVM·HBOS·COPOD(윈도우 피처, PyOD)

입력은 항상 `preprocess.preprocess()` 결과(형식 통일 + 세그먼트 평균 제거)다. 원시값을 넣지 않는다.
- 시계열 모델(`SeqDetector`): 세그먼트별 (n, 3) 배열. 정규화 통계(채널 평균·표준편차)는 train 정상에서만 구한다.
- 피처 모델(`FeatureDetector`): `11_window_features.parquet`의 피처 열.
모든 탐지기는 "클수록 이상"인 윈도우 점수를 낸다. 윈도우는 parquet의 (`seg_uid`, `t_start`, `win_s`)로 지정한다.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

FS = 10.0


# ------------------------------------------------------------------ networks
class CNNNet(nn.Module):
    """DeepAnT: Conv1d ×2 → MaxPool → FC → 다음 샘플 3채널"""
    def __init__(self, lookback, hidden=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(3, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2), nn.Flatten(),
            nn.Linear(hidden * (lookback // 2), 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):                       # x: (B, L, 3)
        return self.body(x.transpose(1, 2)), None


class LSTMNet(nn.Module):
    def __init__(self, lookback, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]), None


class GAT(nn.Module):
    """완전 그래프 위 단일 헤드 graph attention"""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.a = nn.Linear(2 * d, 1, bias=False)

    def forward(self, v):                       # v: (B, N, d)
        h = self.W(v)
        n = h.shape[1]
        pair = torch.cat([h.unsqueeze(2).expand(-1, -1, n, -1), h.unsqueeze(1).expand(-1, n, -1, -1)], -1)
        att = torch.softmax(nn.functional.leaky_relu(self.a(pair).squeeze(-1), 0.2), -1)
        return torch.sigmoid(att @ h)


class MTADGATNet(nn.Module):
    """feature-GAT(노드 = 센서 3개) + time-GAT(노드 = 시점) + GRU → 예측 헤드 + 복원 헤드.
    원 논문의 VAE 복원은 결정적 복원 헤드로 단순화했다."""
    def __init__(self, lookback, hidden=32):
        super().__init__()
        self.L = lookback
        self.feat_gat, self.time_gat = GAT(lookback), GAT(3)
        self.gru = nn.GRU(9, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.rec = nn.Linear(hidden, lookback * 3)

    def forward(self, x):
        hf = self.feat_gat(x.transpose(1, 2)).transpose(1, 2)
        ht = self.time_gat(x)
        h = self.gru(torch.cat([x, hf, ht], -1))[0][:, -1]
        return self.fc(h), self.rec(h).view(-1, self.L, 3)


class GDNNet(nn.Module):
    """센서 임베딩으로 attention 그래프를 만들고 이웃 정보로 각 센서의 다음 값을 예측"""
    def __init__(self, lookback, d=16):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(3, d) * 0.1)
        self.Wx = nn.Linear(lookback, d, bias=False)
        self.a = nn.Linear(4 * d, 1, bias=False)
        self.out = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        z = self.Wx(x.transpose(1, 2))          # (B, 3, d)
        g = torch.cat([self.emb.unsqueeze(0).expand(z.shape[0], -1, -1), z], -1)
        pair = torch.cat([g.unsqueeze(2).expand(-1, -1, 3, -1), g.unsqueeze(1).expand(-1, 3, -1, -1)], -1)
        att = torch.softmax(nn.functional.leaky_relu(self.a(pair).squeeze(-1), 0.2), -1)
        h = torch.relu(att @ z) * self.emb
        return self.out(h).squeeze(-1), None


class SVDDNet(nn.Module):
    def __init__(self, win, rep=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1, bias=False), nn.LeakyReLU(),
            nn.Conv1d(16, 16, 3, padding=1, bias=False), nn.LeakyReLU(),
            nn.MaxPool1d(2), nn.Flatten(), nn.Linear(16 * (win // 2), rep, bias=False))

    def forward(self, x):                       # x: (B, win, 3)
        return self.body(x.transpose(1, 2))


# ------------------------------------------------------------------ training helper
def train_loop(net, loss_fn, tensors, seed, epochs, desc="", progress=True, lr=1e-3, batch=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr)
    n = len(tensors[0])
    bar = tqdm(range(epochs), desc=desc, leave=False, disable=not progress, dynamic_ncols=True)
    for _ in bar:
        net.train()
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net, *[t[idx] for t in tensors])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        bar.set_postfix(loss=f"{tot / n:.4f}")
    return net.eval()


def _win_starts(wins: pd.DataFrame) -> np.ndarray:
    return np.round(wins["t_start"].to_numpy() * FS).astype(int)


# ------------------------------------------------------------------ sequence detectors
class SeqDetector:
    """전처리된 세그먼트 배열을 직접 입력으로 받는 모델의 공통 부분."""
    key = name = family = ""
    ext = "pt"

    def __init__(self, lookback: int = 5, win: int = 10):
        self.lookback, self.win = lookback, win

    def _norm(self, x):
        return ((x - self.mu) / self.sd).astype(np.float32)

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd, "extra": self._extra(),
                    "lookback": self.lookback, "win": self.win}, path)

    @classmethod
    def load(cls, path):
        st = torch.load(path, weights_only=False)
        obj = cls(st["lookback"], st["win"])
        obj.net = obj.build()
        obj.net.load_state_dict(st["state"])
        obj.net.eval()
        obj.mu, obj.sd = st["mu"], st["sd"]
        obj._load_extra(st["extra"])
        return obj

    def _extra(self):
        return {}

    def _load_extra(self, st):
        pass


class ForecastDetector(SeqDetector):
    """lookback 샘플 → 다음 샘플 예측. 윈도우 점수 = 윈도우 안 시점별 점수의 평균
    (시점 점수는 세그먼트 시작 후 lookback 이상 지난 샘플에서만 정의)."""
    robust_max = False   # GDN: 센서별 오차를 median/IQR로 정규화 후 최댓값

    def build(self):
        raise NotImplementedError

    def fit(self, segs: list[np.ndarray], seed=0, epochs=30, progress=True):
        x_all = np.concatenate(segs)
        self.mu, self.sd = x_all.mean(0), x_all.std(0) + 1e-9
        X, Y = [], []
        for x in segs:
            x = self._norm(x)
            for t in range(self.lookback, len(x)):
                X.append(x[t - self.lookback:t])
                Y.append(x[t])
        X, Y = torch.tensor(np.array(X)), torch.tensor(np.array(Y))
        torch.manual_seed(seed)
        self.net = self.build()

        def loss_fn(net, x, y):
            p, r = net(x)
            loss = ((p - y) ** 2).mean()
            return loss + ((r - x) ** 2).mean() if r is not None else loss
        train_loop(self.net, loss_fn, (X, Y), seed, epochs, f"{self.name}", progress)
        with torch.no_grad():
            p, r = self.net(X)
            e = (p - Y).numpy()
        self.stats = {"sd": e.std(0) + 1e-6, "med": np.median(e, 0),
                      "iqr": np.subtract(*np.percentile(e, [75, 25], 0)) + 1e-6,
                      "rsd": ((r - X).numpy().std((0, 1)) + 1e-6) if r is not None else None}
        return self

    def _extra(self):
        return {"stats": self.stats}

    def _load_extra(self, st):
        self.stats = st["stats"]

    def step_scores(self, x_raw: np.ndarray) -> np.ndarray:
        x = self._norm(x_raw)
        s = np.full(len(x), np.nan)
        L = self.lookback
        if len(x) <= L:
            return s
        with torch.no_grad():
            X = torch.tensor(np.stack([x[t - L:t] for t in range(L, len(x))]))
            p, r = self.net(X)
        e = p.numpy() - x[L:]
        if self.robust_max:
            s[L:] = np.max(np.abs(e - self.stats["med"]) / self.stats["iqr"], 1)
        else:
            sc = np.mean((e / self.stats["sd"]) ** 2, 1)
            if r is not None:
                sc = sc + np.mean(((r.numpy() - X.numpy()) / self.stats["rsd"]) ** 2, (1, 2))
            s[L:] = sc
        return s

    def score(self, segs: dict, wins: pd.DataFrame) -> np.ndarray:
        cache = {u: self.step_scores(segs[u]) for u in wins["seg_uid"].unique()}
        starts, w = _win_starts(wins), int(round(wins["win_s"].iloc[0] * FS))
        return np.array([np.nanmean(cache[u][s:s + w]) for u, s in zip(wins["seg_uid"], starts)])


class CNNDeepAnT(ForecastDetector):
    key, name, family = "cnn_deepant", "CNN (DeepAnT)", "Forecasting"

    def build(self):
        return CNNNet(self.lookback)


class LSTMAD(ForecastDetector):
    key, name, family = "lstm_ad", "LSTM-AD", "Forecasting"

    def build(self):
        return LSTMNet(self.lookback)


class MTADGAT(ForecastDetector):
    key, name, family = "mtad_gat", "MTAD-GAT", "Graph"

    def build(self):
        return MTADGATNet(self.lookback)


class GDN(ForecastDetector):
    key, name, family = "gdn", "GDN", "Graph"
    robust_max = True

    def build(self):
        return GDNNet(self.lookback)


class DeepSVDD(SeqDetector):
    """원신호 윈도우(win × 3)를 잠재공간 중심 c 근처로 모으고 거리를 점수로 쓴다."""
    key, name, family = "deep_svdd", "DeepSVDD", "Distribution"

    def build(self):
        return SVDDNet(self.win)

    def fit(self, segs: list[np.ndarray], seed=0, epochs=30, progress=True):
        x_all = np.concatenate(segs)
        self.mu, self.sd = x_all.mean(0), x_all.std(0) + 1e-9
        X = torch.tensor(np.stack([self._norm(x)[s:s + self.win] for x in segs
                                   for s in range(0, len(x) - self.win + 1)]))
        torch.manual_seed(seed)
        self.net = self.build()
        with torch.no_grad():
            self.c = self.net(X).mean(0)
        c = self.c
        train_loop(self.net, lambda net, x: ((net(x) - c) ** 2).sum(1).mean(), (X,), seed, epochs,
                   f"{self.name}", progress)
        return self

    def _extra(self):
        return {"c": self.c}

    def _load_extra(self, st):
        self.c = st["c"]

    def score(self, segs: dict, wins: pd.DataFrame) -> np.ndarray:
        starts = _win_starts(wins)
        X = torch.tensor(np.stack([self._norm(segs[u])[s:s + self.win] for u, s in zip(wins["seg_uid"], starts)]))
        with torch.no_grad():
            return ((self.net(X) - self.c) ** 2).sum(1).numpy()


# ------------------------------------------------------------------ feature detectors (PyOD)
class FeatureDetector:
    """윈도우 피처 표준화 + PyOD 탐지기. `cols`는 parquet 피처 열 이름 목록."""
    key = name = ""
    family = "Distribution"
    ext = "joblib"

    def __init__(self, cols: list[str]):
        self.cols = list(cols)

    def make(self, seed):
        raise NotImplementedError

    def _X(self, F: pd.DataFrame) -> np.ndarray:
        """결측(평탄 윈도우의 kurt·skew, 사인 피팅 실패 등)은 train 중앙값으로 채운다"""
        return F[self.cols].astype(float).fillna(self.fill).to_numpy()

    def fit_features(self, F: pd.DataFrame, seed=0):
        self.fill = F[self.cols].astype(float).median()
        X = self._X(F)
        self.scaler = StandardScaler().fit(X)
        self.det = self.make(seed).fit(self.scaler.transform(X))
        return self

    def score_features(self, F: pd.DataFrame) -> np.ndarray:
        return self.det.decision_function(self.scaler.transform(self._X(F)))

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class MCD(FeatureDetector):
    key, name = "mcd", "MCD"

    def make(self, seed):
        from pyod.models.mcd import MCD as _M
        return _M(random_state=seed)


class OCSVM(FeatureDetector):
    key, name = "ocsvm", "OCSVM"

    def make(self, seed):
        from pyod.models.ocsvm import OCSVM as _M
        return _M(nu=0.05, gamma="scale")


class HBOS(FeatureDetector):
    key, name = "hbos", "HBOS"

    def make(self, seed):
        from pyod.models.hbos import HBOS as _M
        return _M()


class COPOD(FeatureDetector):
    key, name = "copod", "COPOD"

    def make(self, seed):
        from pyod.models.copod import COPOD as _M
        return _M()


SEQ_MODELS = {c.key: c for c in [CNNDeepAnT, LSTMAD, MTADGAT, GDN, DeepSVDD]}
FEATURE_MODELS = {c.key: c for c in [MCD, OCSVM, HBOS, COPOD]}
