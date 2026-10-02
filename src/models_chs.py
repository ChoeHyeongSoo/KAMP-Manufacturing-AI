"""주제 ③ — 27 CNN+LSTM 오토인코더(복원 계열) 및 선형 대조군 (노트북 `27_model_cnn_lstm_ae_CHS`).

설계 의도
- JIW 24·25·26이 예측(다음 샘플)·그래프·분포 계열이라면, 여기서는 **평가 윈도우와 같은 원신호 윈도우 전체를 복원**한다.
  Conv1d 인코더(국소 형상) → LSTM(시간 문맥) → 잠재 z → LSTM 디코더 → 채널별 복원, 점수 = 정규화한 복원 오차.
- 점수를 채널별 e_c로 분해해 저장한다(어느 센서가 이상을 만들었는가: 심사 3번 FN·FP 원인 설명, 5번 채널 불일치).
- 선형 대조군 `PCARawAE`(같은 윈도우 벡터에 PCA 복원)를 두어 비선형·시간 구조의 이득을 따로 본다.
- 평가 계약은 JIW와 같다: `benchmark_jiw.iter_splits`(group_kfold_seg · time_block), 임계 = train 정상 점수 q99,
  `evaluate` 지표. 정규화·임계·PCA는 모두 train 정상에서만 적합한다(비지도라 라벨로 하이퍼파라미터를 고르지 않는다).

학습량(epochs 200, batch 64, lr 1e-3)은 train·monitor 정상 손실의 수렴(라벨 미사용)으로 정했다: ep40·b256은 MSE 0.53에서 하강 중, ep200·b64에서 평탄.

금지 입력: 원시값(형식 잔여 신호)·DC·절대 시간·`state`·`label`·`src`. 입력은 `preprocess.preprocess()` 결과 윈도우뿐이다.
`all3+res`(세그먼트 사인 피팅 잔차 채널)는 세그먼트 전체 피팅이라 날짜 교란 가능성이 있는 ablation 전용 변형이다.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

import benchmark_jiw as B
import evaluate as ev
import paths
import signal_checks

FS = 10.0
VARIANTS = ["all3", "vib2", "cur1", "all3+res", "all3_zwin"]
CH_NAMES = {"all3": ["AI0", "AI1", "AI2"], "vib2": ["AI0", "AI1"], "cur1": ["AI2"],
            "all3+res": ["AI0", "AI1", "AI2", "AI2res"], "all3_zwin": ["AI0", "AI1", "AI2"]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def variant_tag(variant: str) -> str:
    """모델 이름용 변형 표기(`+` 제거): all3+res → all3_res"""
    return variant.replace("+", "_")


# ------------------------------------------------------------------ network
class CNNLSTMAENet(nn.Module):
    """Conv1d ×2 → LSTM(마지막 은닉) → z(d) → z 반복 → LSTM → Linear. 입력 (B, W, C)."""

    def __init__(self, win: int, n_channels: int, d: int = 8, hidden: int = 32):
        super().__init__()
        self.win = win
        self.conv = nn.Sequential(nn.Conv1d(n_channels, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(16, 32, 3, padding=1), nn.ReLU())
        self.enc = nn.LSTM(32, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, d)
        self.dec = nn.LSTM(d, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_channels)

    def encode(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)      # (B, W, 32)
        return self.to_z(self.enc(h)[0][:, -1])               # (B, d)

    def forward(self, x):
        z = self.encode(x)
        return self.out(self.dec(z.unsqueeze(1).expand(-1, self.win, -1))[0])


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# ------------------------------------------------------------------ detectors
class _WindowAE:
    """윈도우 복원 탐지기 공통부: 채널 정규화(train 정상 전체), 채널별 복원 오차 정규화, 점수."""
    name = ""

    def _norm(self, x):
        return ((x - self.mu) / self.sd).astype(np.float32)

    def _recon(self, xn: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _finish_fit(self, xn: np.ndarray):
        """train 복원 잔차의 채널별 sd(σ_c)를 저장한다."""
        r = self._recon(xn) - xn
        self.sigma = (r.std((0, 1)) + 1e-6).astype(np.float32)

    def channel_scores(self, windows: np.ndarray) -> np.ndarray:
        xn = self._norm(windows)
        r = self._recon(xn) - xn
        return ((r / self.sigma) ** 2).mean(1)                # (N, C)

    def score(self, windows: np.ndarray) -> np.ndarray:
        return self.channel_scores(windows).mean(1)


class CNNLSTMAE(_WindowAE):
    """CNN+LSTM 오토인코더. SeqDetector 규약 호환(fit/score/channel_scores/save/load), 입력은 (N, W, C) 윈도우."""
    name = "cnn_lstm_ae"

    def __init__(self, win: int, n_channels: int, d: int = 8, epochs: int = 200, lr: float = 1e-3,
                 batch: int = 64, seed: int = 0):
        self.win, self.C, self.d, self.epochs, self.lr, self.batch, self.seed = win, n_channels, d, epochs, lr, batch, seed
        self.history: list[dict] = []
        self.net = None

    def build(self):
        return CNNLSTMAENet(self.win, self.C, self.d)

    def fit(self, train_windows: np.ndarray, groups=None, monitor_frac: float = 0.1):
        """train 정상 윈도우로 학습. groups(seg_uid)가 있으면 세그먼트 단위 10%를 손실 모니터링용으로 떼어 둔다
        (경사 갱신에는 쓰지 않고, 모델·epoch 선택에도 쓰지 않는다 — 과적합 확인용 곡선만)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        x = np.asarray(train_windows, dtype=np.float32)
        self.mu = x.reshape(-1, self.C).mean(0)
        self.sd = x.reshape(-1, self.C).std(0) + 1e-9
        xn = self._norm(x)
        mon = np.zeros(len(xn), bool)
        if groups is not None:
            g = np.asarray(groups)
            uniq = np.unique(g)
            rng = np.random.default_rng(self.seed)
            pick = rng.choice(uniq, max(1, int(round(len(uniq) * monitor_frac))), replace=False)
            mon = np.isin(g, pick)
        Xtr = torch.tensor(xn[~mon]).to(DEVICE)
        Xmo = torch.tensor(xn[mon]).to(DEVICE) if mon.any() else None
        self.net = self.build().to(DEVICE)
        opt = torch.optim.Adam(self.net.parameters(), self.lr)
        n = len(Xtr)
        self.history = []
        t_fit = time.time()
        for ep in range(self.epochs):
            self.net.train()
            perm, tot = torch.randperm(n), 0.0
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch].to(DEVICE)
                opt.zero_grad()
                loss = ((self.net(Xtr[idx]) - Xtr[idx]) ** 2).mean()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            rec = {"epoch": ep + 1, "train": tot / n, "monitor": np.nan}
            if Xmo is not None:
                self.net.eval()
                with torch.no_grad():
                    rec["monitor"] = float(((self.net(Xmo) - Xmo) ** 2).mean())
            self.history.append(rec)
        self.net.eval()
        self._finish_fit(xn)
        self.train_sec = round(time.time() - t_fit, 1)
        return self

    def _recon(self, xn: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xn), 1024):
                out.append(self.net(torch.tensor(xn[i:i + 1024]).to(DEVICE)).cpu().numpy())
        return np.concatenate(out)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """잠재 벡터 z (N, d)"""
        xn = self._norm(windows)
        with torch.no_grad():
            return self.net.encode(torch.tensor(xn).to(DEVICE)).cpu().numpy()

    def n_params(self) -> int:
        return count_params(self.net)

    def save(self, path):
        torch.save({"state": {k: v.cpu() for k, v in self.net.state_dict().items()}, "mu": self.mu, "sd": self.sd,
                    "sigma": self.sigma, "history": self.history, "train_sec": getattr(self, "train_sec", np.nan),
                    "cfg": dict(win=self.win, n_channels=self.C, d=self.d, epochs=self.epochs, lr=self.lr,
                                batch=self.batch, seed=self.seed)}, path)

    @classmethod
    def load(cls, path):
        st = torch.load(path, weights_only=False, map_location="cpu")
        obj = cls(**st["cfg"])
        obj.net = obj.build().to(DEVICE)
        obj.net.load_state_dict(st["state"])
        obj.net.eval()
        obj.mu, obj.sd, obj.sigma, obj.history = st["mu"], st["sd"], st["sigma"], st["history"]
        obj.train_sec = st.get("train_sec", np.nan)
        return obj


class PCARawAE(_WindowAE):
    """선형 대조군: 윈도우 벡터(W·C, 채널 정규화 후)에 PCA(n_components=d)를 적합하고 복원 오차를 점수로 쓴다."""
    name = "pca_raw_ae"

    def __init__(self, win: int, n_channels: int, d: int = 8, seed: int = 0, **_):
        self.win, self.C, self.d, self.seed = win, n_channels, d, seed
        self.history: list[dict] = []

    def fit(self, train_windows: np.ndarray, groups=None, **_):
        x = np.asarray(train_windows, dtype=np.float32)
        self.mu = x.reshape(-1, self.C).mean(0)
        self.sd = x.reshape(-1, self.C).std(0) + 1e-9
        xn = self._norm(x)
        t_fit = time.time()
        self.pca = PCA(n_components=self.d, svd_solver="full", random_state=self.seed).fit(xn.reshape(len(xn), -1))
        self._finish_fit(xn)
        self.train_sec = round(time.time() - t_fit, 1)
        return self

    def _recon(self, xn: np.ndarray) -> np.ndarray:
        flat = xn.reshape(len(xn), -1)
        rec = self.pca.inverse_transform(self.pca.transform(flat))
        return rec.reshape(xn.shape).astype(np.float32)

    def n_params(self) -> int:
        return int(self.pca.components_.size)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


MODELS = {"cnn_lstm_ae": CNNLSTMAE, "pca_raw_ae": PCARawAE}


# ------------------------------------------------------------------ 변형 입력
def _sine_residual(x: np.ndarray) -> np.ndarray:
    """세그먼트 AI2 − 사인 피팅값. 피팅 실패(길이 < 4, 파라미터 nan)면 원신호."""
    fit = signal_checks.sine_fit_segment(x, FS)
    if not np.isfinite(fit["f_hz"]):
        return x.astype(np.float32)
    t = np.arange(len(x)) / FS
    return (x - (fit["amp"] * np.sin(2 * np.pi * fit["f_hz"] * t + fit["phase"]) + fit["offset"])).astype(np.float32)


def make_variant_segments(segs: dict, variant: str) -> dict:
    """세그먼트 (n, 3) → variant별 (n, C). `all3_zwin`은 윈도우 단위 처리라 여기서는 all3와 같고 `extract_windows(zwin=True)`에서 적용한다."""
    if variant in ("all3", "all3_zwin"):
        return {u: x for u, x in segs.items()}
    if variant == "vib2":
        return {u: x[:, :2] for u, x in segs.items()}
    if variant == "cur1":
        return {u: x[:, 2:3] for u, x in segs.items()}
    if variant == "all3+res":
        return {u: np.column_stack([x, _sine_residual(x[:, 2])]).astype(np.float32) for u, x in segs.items()}
    raise ValueError(variant)


def extract_windows(segs: dict, wins: pd.DataFrame, win: int, zwin: bool = False) -> np.ndarray:
    """parquet 윈도우 행(seg_uid, t_start) → (N, W, C). `benchmark_jiw.recompute_features`와 같은 절단 규약.
    zwin이면 윈도우마다 채널별 z-score(표준편차 0이면 0)."""
    out = []
    for u, t in zip(wins["seg_uid"], wins["t_start"]):
        s = int(round(t * FS))
        out.append(segs[u][s:s + win])
    X = np.stack(out).astype(np.float32)
    if zwin:
        m, sd = X.mean(1, keepdims=True), X.std(1, keepdims=True)
        X = np.where(sd > 0, (X - m) / np.where(sd > 0, sd, 1), 0.0).astype(np.float32)
    return X


# ------------------------------------------------------------------ 실행
def run_variant(segs: dict, W: pd.DataFrame, variant: str, win_s: float, model: str = "cnn_lstm_ae",
                seeds=(0, 1, 2), d: int = 8, epochs: int = 200, retrain: bool = False, nb: str = "27_model_cnn_lstm_ae_CHS",
                splits=None, progress: bool = True, with_train: bool = False):
    """한 variant·윈도우 길이에 대해 9개 분할 × seeds를 학습·평가한다.

    반환 (cv_rows, score_rows).
    - cv_rows: 분할·seed당 1행(지표, 임계, 상태별 오경보율, 학습 초, 파라미터 수).
    - score_rows: 분할·seed·test 윈도우당 1행(점수, 채널별 e_c, 임계, 초과 여부). 학습 윈도우 점수는 저장하지 않는다.
    - with_train=True면 train 정상 윈도우 점수 행도 `part='train'`으로 덧붙인다(세그먼트 집계 임계용; 가중치를 불러오면 재학습 없이 가능).
    - 가중치: `models/<nb>/<model_id>_<split>_f<fold>_s<seed>.(pt|joblib)`. `retrain=False`면 있는 것을 불러온다.
    """
    from tqdm.auto import tqdm
    win = int(round(win_s * FS))
    cls = MODELS[model]
    model_id = f"{model}_{variant_tag(variant)}_w{win_s:g}s"
    ch = CH_NAMES[variant]
    vs = make_variant_segments(segs, variant)
    zwin = variant == "all3_zwin"
    Wv = W[W["win_s"] == win_s].reset_index(drop=True)
    jobs = [(sp, k, tr, te, s) for sp, k, tr, te in B.iter_splits(Wv) if splits is None or sp in splits for s in seeds]
    cv, sc = [], []
    for sp, k, tr, te, seed in tqdm(jobs, desc=model_id, disable=not progress, leave=False):
        ext = "pt" if model == "cnn_lstm_ae" else "joblib"
        path = paths.MODELS / nb / f"{model_id}_{sp}_f{k}_s{seed}.{ext}"
        Xtr = extract_windows(vs, tr, win, zwin)
        Xte = extract_windows(vs, te, win, zwin)
        t0 = time.time()
        if path.exists() and not retrain:
            m = cls.load(path)
            status = "loaded"
        else:
            m = cls(win, len(ch), d=d, seed=seed) if model == "pca_raw_ae" else cls(win, len(ch), d=d, epochs=epochs, seed=seed)
            m.fit(Xtr, groups=tr["seg_uid"].to_numpy())
            path.parent.mkdir(parents=True, exist_ok=True)
            m.save(path)
            status = "trained"
        sec = round(time.time() - t0, 1) if status == "trained" else float(getattr(m, "train_sec", np.nan))   # 불러온 경우 저장된 학습 초
        s_tr = m.score(Xtr)
        E = m.channel_scores(Xte)
        s_te = E.mean(1)
        thr = ev.threshold_from_normal(s_tr)
        y = te["label"].to_numpy()
        n_te, o_te = te[y == 0], te[y == 1]
        sn, so = s_te[y == 0], s_te[y == 1]
        dly = ev.detection_delay_s(so, o_te["seg_uid"], o_te["t_start"], thr, win_s=win_s)
        felt = ev.detection_delay_s(so, o_te["seg_uid"], o_te["t_start"], thr, win_s=win_s, include_gap=True)
        det = dly["detected"].to_numpy(bool)
        st = n_te["state"].to_numpy()
        cv.append({"model": model, "variant": variant, "model_id": model_id, "win_s": win_s, "split": sp, "fold": k,
                   "seed": seed, "status": status, "auc": ev.auc(y, s_te), "fpr_sample": ev.fpr_sample(sn, thr),
                   "fpr_segment": ev.fpr_segment(sn, n_te["seg_uid"], thr),
                   "delay_median_s": float(np.nanmedian(dly["delay_in_window_s"])) if det.any() else np.nan,
                   "felt_delay_median_s": float(np.nanmedian(felt["delay_s"])) if det.any() else np.nan,
                   "thr": thr, "n_out_seg": len(dly), "n_detected": int(det.sum()),
                   "fpr_high_load": float(np.mean(sn[st == 0] > thr)) if (st == 0).any() else np.nan,
                   "fpr_low_load": float(np.mean(sn[st == 1] > thr)) if (st == 1).any() else np.nan,
                   "n_params": m.n_params(), "sec": sec})
        df = pd.DataFrame({"seg_uid": te["seg_uid"].to_numpy(), "t_start": te["t_start"].to_numpy(), "win_s": win_s,
                           "split": sp, "fold": k, "seed": seed, "label": y, "state": te["state"].to_numpy(),
                           "model": model_id, "score": s_te, "thr": thr, "exceed": s_te > thr, "part": "test"})
        for j, c in enumerate(ch):
            df[f"e_{c}"] = E[:, j]
        sc.append(df)
        if with_train:
            sc.append(pd.DataFrame({"seg_uid": tr["seg_uid"].to_numpy(), "t_start": tr["t_start"].to_numpy(),
                                    "win_s": win_s, "split": sp, "fold": k, "seed": seed, "label": 0,
                                    "state": tr["state"].to_numpy(), "model": model_id, "score": s_tr, "thr": thr,
                                    "exceed": s_tr > thr, "part": "train"}))
        if k == 0 and sp == "group_kfold_seg" and seed == seeds[0] and getattr(m, "history", None):
            pd.DataFrame(m.history).assign(model_id=model_id).to_csv(
                paths.MODELS / nb / f"{model_id}_history.csv", index=False)
    return pd.DataFrame(cv), pd.concat(sc, ignore_index=True)
