"""주제 ③ 프레스 유압펌프 — 고전 시계열·트리 이상탐지 모델.

22_model_classical_timeseries_JSC에서 사용하는 네 모델을 정의한다.

- Naive persistence: 직전 관측을 다음 관측의 예측값으로 사용
- Ridge-VAR: 과거 p개 3채널 값을 선형 결합해 다음 3채널을 예측
- Kalman innovation: VAR(1) 상태전이 + 선형 Kalman filter의 innovation 거리
- Isolation Forest: 1초 윈도우의 진폭 피처 18열을 트리로 고립

시계열 모델은 세그먼트 경계를 넘지 않는다. 채널 정규화·잔차 척도·Ridge alpha·
Kalman 공분산은 모두 학습 정상 데이터에서만 적합한다. 점수는 클수록 이상이다.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FS = 10.0
EPS = 1e-6


def _pairs(segs: list[np.ndarray], lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """세그먼트 경계를 넘지 않는 과거 lookback → 다음 샘플 학습쌍."""
    xs, ys = [], []
    for x in segs:
        if len(x) <= lookback:
            continue
        for t in range(lookback, len(x)):
            xs.append(x[t - lookback:t].reshape(-1))
            ys.append(x[t])
    if not xs:
        raise ValueError(f"lookback={lookback}보다 긴 학습 세그먼트가 없다")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _regularized_cov(x: np.ndarray, floor: float = EPS) -> np.ndarray:
    """작은 표본에서도 역행렬이 안정적인 대칭 공분산."""
    c = np.atleast_2d(np.cov(np.asarray(x, dtype=float), rowvar=False))
    c = (c + c.T) / 2
    scale = max(float(np.trace(c)) / max(len(c), 1), 1.0)
    return c + np.eye(len(c)) * floor * scale


class ClassicalSeqDetector:
    """benchmark_jiw._Runner와 호환되는 비신경망 시계열 탐지기."""

    key = name = ""
    family = "Classical forecasting"
    ext = "joblib"

    def __init__(self, lookback: int = 5, win: int = 10):
        self.lookback, self.win = int(lookback), int(win)

    def _fit_scale(self, segs: list[np.ndarray]) -> list[np.ndarray]:
        x = np.concatenate(segs).astype(np.float64)
        self.mu = x.mean(axis=0)
        self.sd = np.maximum(x.std(axis=0), EPS)
        return [self._norm(s) for s in segs]

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sd

    def fit(self, segs: list[np.ndarray], seed=0, epochs=None, progress=False):
        raise NotImplementedError

    def step_scores(self, x_raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, segs: dict[str, np.ndarray], wins: pd.DataFrame) -> np.ndarray:
        """시점 점수를 윈도우 안에서 평균한다(기존 24 예측 모델과 같은 집계)."""
        cache = {u: self.step_scores(segs[u]) for u in wins["seg_uid"].unique()}
        starts = np.round(wins["t_start"].to_numpy(dtype=float) * FS).astype(int)
        win = int(round(float(wins["win_s"].iloc[0]) * FS))
        out = []
        for uid, start in zip(wins["seg_uid"], starts):
            v = cache[uid][start:start + win]
            out.append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
        return np.asarray(out)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def details(self) -> dict:
        return {"lookback": self.lookback}


class NaivePersistence(ClassicalSeqDetector):
    key, name = "naive_persistence", "Naive persistence"

    def __init__(self, lookback: int = 5, win: int = 10):
        # benchmark의 공통 lookback 인자와 무관하게 정의상 직전 1개만 사용한다.
        super().__init__(1, win)

    def fit(self, segs: list[np.ndarray], seed=0, epochs=None, progress=False):
        z = self._fit_scale(segs)
        err = np.concatenate([x[1:] - x[:-1] for x in z if len(x) > 1])
        self.err_sd = np.maximum(err.std(axis=0), EPS)
        return self

    def step_scores(self, x_raw: np.ndarray) -> np.ndarray:
        x = self._norm(x_raw)
        s = np.full(len(x), np.nan)
        if len(x) > 1:
            e = (x[1:] - x[:-1]) / self.err_sd
            s[1:] = np.mean(e ** 2, axis=1)
        return s

    def details(self) -> dict:
        return {"lookback": 1, **{f"resid_sd_{i}": v for i, v in enumerate(self.err_sd)}}


class RidgeVAR(ClassicalSeqDetector):
    key, name = "ridge_var", "Ridge-VAR"
    ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

    @staticmethod
    def _inner_split(segs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """세그먼트 순서 뒤 20%를 정상 validation으로 둔다."""
        n_val = max(1, int(round(len(segs) * 0.2)))
        if len(segs) - n_val < 2:
            return segs, segs
        return segs[:-n_val], segs[-n_val:]

    def fit(self, segs: list[np.ndarray], seed=0, epochs=None, progress=False):
        fit_raw, val_raw = self._inner_split(segs)
        # alpha 선택용 스케일도 내부 fit 정상에서만 계산한다(validation 통계 선사용 방지).
        fit_all = np.concatenate(fit_raw).astype(np.float64)
        inner_mu = fit_all.mean(axis=0)
        inner_sd = np.maximum(fit_all.std(axis=0), EPS)
        fit_segs = [(np.asarray(x, dtype=np.float64) - inner_mu) / inner_sd for x in fit_raw]
        val_segs = [(np.asarray(x, dtype=np.float64) - inner_mu) / inner_sd for x in val_raw]
        x_fit, y_fit = _pairs(fit_segs, self.lookback)
        x_val, y_val = _pairs(val_segs, self.lookback)
        losses = {}
        for alpha in self.ALPHAS:
            m = Ridge(alpha=alpha).fit(x_fit, y_fit)
            losses[alpha] = float(np.mean((m.predict(x_val) - y_val) ** 2))
        self.alpha = min(losses, key=lambda a: (losses[a], a))
        self.val_mse_by_alpha = losses
        # 외부 fold의 최종 모델은 alpha 고정 후 전체 학습 정상으로 다시 적합한다.
        z = self._fit_scale(segs)
        x_all, y_all = _pairs(z, self.lookback)
        self.model = Ridge(alpha=self.alpha).fit(x_all, y_all)
        self.err_sd = np.maximum((self.model.predict(x_all) - y_all).std(axis=0), EPS)
        return self

    def step_scores(self, x_raw: np.ndarray) -> np.ndarray:
        x = self._norm(x_raw)
        s = np.full(len(x), np.nan)
        if len(x) > self.lookback:
            xx, yy = _pairs([x], self.lookback)
            e = (self.model.predict(xx) - yy) / self.err_sd
            s[self.lookback:] = np.mean(e ** 2, axis=1)
        return s

    def details(self) -> dict:
        return {
            "lookback": self.lookback,
            "selected_alpha": self.alpha,
            "validation_mse": self.val_mse_by_alpha[self.alpha],
            "coef_l2": float(np.linalg.norm(self.model.coef_)),
        }


class KalmanInnovation(ClassicalSeqDetector):
    key, name = "kalman_innovation", "Kalman innovation"

    def fit(self, segs: list[np.ndarray], seed=0, epochs=None, progress=False):
        z = self._fit_scale(segs)
        x, y = _pairs(z, 1)
        transition = Ridge(alpha=0.1).fit(x, y)
        # sklearn 다중 출력: y = x @ coef_.T + intercept
        self.F = transition.coef_.astype(np.float64)
        self.b = transition.intercept_.astype(np.float64)
        resid = y - transition.predict(x)
        self.Q = _regularized_cov(resid)
        # 센서 관측 잡음은 동역학 잔차 분산의 10%로 고정한다. 이상 라벨로 조정하지 않는다.
        self.R = np.diag(np.maximum(np.diag(self.Q) * 0.1, EPS))
        self.P0 = _regularized_cov(np.concatenate(z))
        self.spectral_radius = float(np.max(np.abs(np.linalg.eigvals(self.F))))
        return self

    def step_scores(self, x_raw: np.ndarray) -> np.ndarray:
        x = self._norm(x_raw)
        s = np.full(len(x), np.nan)
        if len(x) <= 1:
            return s
        state = x[0].copy()
        p = self.P0.copy()
        eye = np.eye(x.shape[1])
        for t in range(1, len(x)):
            pred = self.F @ state + self.b
            p_pred = self.F @ p @ self.F.T + self.Q
            innovation = x[t] - pred
            innovation_cov = (p_pred + self.R)
            innovation_cov = (innovation_cov + innovation_cov.T) / 2
            inv_s = np.linalg.pinv(innovation_cov)
            s[t] = float(innovation @ inv_s @ innovation)
            gain = p_pred @ inv_s
            state = pred + gain @ innovation
            p = (eye - gain) @ p_pred
            p = (p + p.T) / 2
        return s

    def details(self) -> dict:
        return {
            "lookback": 1,
            "spectral_radius": self.spectral_radius,
            "q_trace": float(np.trace(self.Q)),
            "r_trace": float(np.trace(self.R)),
        }


class IsolationForestAmp:
    key, name = "iforest", "Isolation Forest"
    family = "Distribution"
    ext = "joblib"

    def __init__(self, cols: list[str]):
        self.cols = list(cols)

    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.cols].astype(float).fillna(self.fill).to_numpy()

    def make(self, seed):
        return IsolationForest(
            n_estimators=300,
            max_samples="auto",
            max_features=1.0,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )

    def fit_features(self, frame: pd.DataFrame, seed=0):
        self.fill = frame[self.cols].astype(float).median()
        x = self._x(frame)
        self.scaler = StandardScaler().fit(x)
        self.det = self.make(seed).fit(self.scaler.transform(x))
        return self

    def score_features(self, frame: pd.DataFrame) -> np.ndarray:
        # sklearn score_samples는 클수록 정상이라 부호를 뒤집는다.
        return -self.det.score_samples(self.scaler.transform(self._x(frame)))

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def details(self) -> dict:
        return {
            "n_estimators": self.det.n_estimators,
            "max_samples_fitted": self.det.max_samples_,
            "n_features": len(self.cols),
        }


SEQ_MODELS = {c.key: c for c in (NaivePersistence, RidgeVAR, KalmanInnovation)}
FEATURE_MODELS = {IsolationForestAmp.key: IsolationForestAmp}


_MODEL_FILE = re.compile(
    r"(?P<model>.+)_w(?P<win>[0-9.]+)s_(?P<split>group_kfold_seg|time_block)_f(?P<fold>\d+)_s(?P<seed>\d+)\.joblib$"
)


def collect_model_details(model_dir: Path) -> pd.DataFrame:
    """git 제외 가중치에서 fold별 선택값·상태공간 진단값을 표로 모은다."""
    rows = []
    for path in sorted(Path(model_dir).glob("*.joblib")):
        match = _MODEL_FILE.match(path.name)
        if not match:
            continue
        obj = joblib.load(path)
        row = match.groupdict()
        row.update({"fold": int(row["fold"]), "seed": int(row["seed"]), "win_s": float(row.pop("win"))})
        if hasattr(obj, "details"):
            row.update(obj.details())
        rows.append(row)
    return pd.DataFrame(rows)
