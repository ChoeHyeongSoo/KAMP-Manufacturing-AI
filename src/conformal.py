"""분할 컨포멀(split conformal) p-value 공통 함수.

이상 점수("클수록 이상")를 **정상 데이터만으로** 보정된 p-value로 바꾼다. 학습 정상을 시간순으로
proper-train(모델 적합)과 calibration(점수 분포 기준)으로 나누고, 새 점수 s가 calibration 점수 n개 사이에서
얼마나 극단적인지를 센다.

    p(s) = (1 + #{calibration 점수 >= s}) / (n + 1)

보장: calibration 점수와 새 정상 점수가 교환 가능(exchangeable)하면 P(p <= alpha | 정상) <= alpha.
p는 "정상이라는 가정 아래 이만큼 극단적일 확률의 상한"이며 이상일 확률이 아니다(1 − p를 이상 확률로 읽지 않는다).
교환 가능성이 깨지면(분포 이동, 한 burst 안에서 겹치는 윈도우의 의존) 보장도 깨진다.

스트리밍 규칙: `p <= alpha`는 `s > conformal_threshold(calibration 점수, alpha)`와 같다(동점 포함). 따라서 운영에서는
p-value를 매번 계산하지 않고 임계 하나만 들고 있으면 된다. 이상 데이터는 어떤 함수에도 넣지 않는다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_EPS = 1e-9   # (n + 1) * alpha 가 정수일 때 부동소수 오차로 등급이 한 칸 밀리지 않게 하는 여유


def _clean(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=float).ravel()
    if np.isnan(s).any():
        raise ValueError("점수에 NaN이 있다 — 컨포멀 순위를 정의할 수 없다")
    return s


def segment_order(seg_uid: str) -> int:
    """`normal_123` → 123. 세그먼트 시간 순서는 seg_uid의 숫자 접미사다."""
    return int(str(seg_uid).rsplit("_", 1)[-1])


def time_ordered_split(seg_uids, cal_frac: float) -> tuple[list[str], list[str]]:
    """학습 정상 세그먼트를 시간순으로 정렬해 (proper-train, calibration)으로 나눈다.

    가장 최근 `cal_frac` 비율을 calibration으로 둔다(운영 절차의 "최신 정상 calibration block으로 임계 재고정"과 같은 방향).
    개수 규칙: n_cal = max(1, floor(cal_frac * n + 0.5)) — 반올림(0.5는 올림), 최소 1개. proper-train이 1개 미만이 되면 ValueError.
    입력에 중복이 있으면 첫 등장만 쓴다.
    """
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f"cal_frac은 (0, 1) 범위여야 한다: {cal_frac}")
    uids = sorted(dict.fromkeys(str(u) for u in seg_uids), key=segment_order)
    n = len(uids)
    n_cal = max(1, int(math.floor(cal_frac * n + 0.5)))
    if n - n_cal < 1:
        raise ValueError(f"세그먼트 {n}개로는 proper-train을 남길 수 없다")
    return uids[: n - n_cal], uids[n - n_cal:]


def random_split(seg_uids, cal_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """학습 정상 세그먼트에서 calibration을 **무작위로** 뽑아 (proper-train, calibration)으로 나눈다.

    `time_ordered_split`과 같은 개수 규칙을 쓰고, 두 목록은 시간순으로 정렬해 돌려준다.
    시간순 분할의 calibration이 운전 상태 구성에서 test와 다를 때, 그 영향을 분리해 보는 대조용이다.
    """
    if not 0.0 < cal_frac < 1.0:
        raise ValueError(f"cal_frac은 (0, 1) 범위여야 한다: {cal_frac}")
    uids = sorted(dict.fromkeys(str(u) for u in seg_uids), key=segment_order)
    n = len(uids)
    n_cal = max(1, int(math.floor(cal_frac * n + 0.5)))
    if n - n_cal < 1:
        raise ValueError(f"세그먼트 {n}개로는 proper-train을 남길 수 없다")
    pick = set(np.random.default_rng(seed).choice(n, size=n_cal, replace=False).tolist())
    return [u for i, u in enumerate(uids) if i not in pick], [u for i, u in enumerate(uids) if i in pick]


def min_p(n_cal: int) -> float:
    """calibration 점수 n개로 얻을 수 있는 가장 작은 p-value = 1 / (n + 1)."""
    return 1.0 / (int(n_cal) + 1)


def n_allowed(n_cal: int, alpha: float) -> int:
    """`p <= alpha`가 되려면 넘어서야 하는 calibration 점수 개수의 여유: floor((n + 1) * alpha).

    새 점수 이상인 calibration 점수가 `n_allowed − 1`개 이하일 때 p <= alpha이다. 0이면 그 alpha는 도달 불가다.
    """
    return int(math.floor((int(n_cal) + 1) * float(alpha) + _EPS))


def is_attainable(n_cal: int, alpha: float) -> bool:
    """alpha 수준의 기각이 가능한가: 1 / (n + 1) <= alpha."""
    return n_allowed(n_cal, alpha) >= 1


def p_values(cal_scores, test_scores) -> np.ndarray:
    """컨포멀 p-value. p = (1 + #{calibration 점수 >= 점수}) / (n + 1), 범위 [1/(n+1), 1].

    동점은 보수적으로 센다(calibration 점수와 같으면 '이상 쪽'으로 세지 않는다 = p가 커진다).
    """
    cal = np.sort(_clean(cal_scores))
    if len(cal) == 0:
        raise ValueError("calibration 점수가 비었다")
    s = _clean(test_scores)
    n_ge = len(cal) - np.searchsorted(cal, s, side="left")
    return (1.0 + n_ge) / (len(cal) + 1.0)


def conformal_threshold(cal_scores, alpha: float) -> float:
    """수준 alpha의 컨포멀 임계: calibration 점수 중 ceil((n + 1)(1 − alpha))번째로 작은 값.

    그 순위가 n을 넘으면(= alpha < 1/(n+1), 도달 불가) 무한대를 돌려준다 — 어떤 점수도 경보가 되지 않는다.
    `점수 > 임계`는 `p_values(...) <= alpha`와 정확히 같다. 순위는 n + 1 − floor((n + 1) * alpha)로 계산한다
    (ceil((n + 1)(1 − alpha))와 같은 정수이고 부동소수 오차에 덜 민감하다).
    """
    cal = np.sort(_clean(cal_scores))
    n = len(cal)
    if n == 0:
        raise ValueError("calibration 점수가 비었다")
    rank = n + 1 - n_allowed(n, alpha)          # 1-기준 순위
    if rank > n:
        return float("inf")
    return float(cal[max(rank, 1) - 1])


def reject(p, alpha: float) -> np.ndarray:
    """`p <= alpha` 판정(부동소수 여유 포함). p-value는 1/(n+1)의 배수라 여유가 판정을 바꾸지 않는다."""
    return np.asarray(p, dtype=float) <= float(alpha) + _EPS * 1e-3


def segment_max(scores, seg_uid) -> pd.Series:
    """윈도우 점수를 세그먼트(burst)별 최댓값으로 집계한다. 순서는 seg_uid가 처음 등장한 순서."""
    t = pd.DataFrame({"s": _clean(scores), "u": np.asarray(seg_uid)})
    return t.groupby("u", sort=False)["s"].max()


def binomial_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper–Pearson(정확) 이항 신뢰구간. n = 0이면 (nan, nan)."""
    from scipy.stats import beta

    k, n = int(k), int(n)
    if n == 0:
        return float("nan"), float("nan")
    a = (1.0 - conf) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi
