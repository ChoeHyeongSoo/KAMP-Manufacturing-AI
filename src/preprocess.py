"""주제 ③ 프레스 유압펌프 — 공통 전처리 (형식 통일 · DC 오프셋 제거 · 형식 피처).

02 진단(reports/02_diagnosis_deep.md §2, §9)에서 확인한 "파일 형식 누수"를 모델 입력 전에 차단한다.

- 정상/이상 파일은 저장 경로가 달라 (a) 진동 소수 자릿수(6 vs 8자리), (b) 전류값의 1.19209 격자
  정수배 비율(0.19% vs 99.67%), (c) 세그먼트 DC 오프셋(|DC| 단독 AUC 0.89~0.94)만으로 파일(=라벨)이
  구분된다. 이 셋은 물리 신호가 아니라 표기·저장 형식이므로 반드시 제거하고 나서 피처를 만든다.
- `unify_format` → `remove_dc` 순서로 적용한다. 형식 통일이 먼저여야 반올림·양자화가 DC 제거 결과를
  다시 깨뜨리지 않는다.
- `format_features`는 "형식 피처만" 뽑는 진단용이다. 전처리 전/후 이 피처만으로 파일을 구분할 수 있는지
  (03 노트북 A-3 차단 실험)를 볼 때 쓰며, 모델 입력으로는 쓰지 않는다.

입력 프레임은 `data_quality.load(key)` 결과(샘플 단위, `src`·`seg` 컬럼 포함) 또는 그것을
`pd.concat`한 것이다. 세그먼트 키는 `seg_uid`(= `src_seg`)이며 없으면 `add_seg_uid`로 만든다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VIB_CHANNELS = ["AI0_Vibration", "AI1_Vibration"]
CUR_CHANNEL = "AI2_Current"
SENSORS = VIB_CHANNELS + [CUR_CHANNEL]

VIB_DECIMALS = 6          # 정상 파일 진동 소수 자릿수(다수 형식) — 02 §2
CUR_STEP = 1.19209        # 이상 파일 전류 양자화 격자 — 02 §2 (float32 eps × 1e7로 추정)
CUR_STEP_TOL = 1e-3       # 격자 정수배 판정 허용 오차 (signal_checks.quantization_multiple_share와 동일)

DC_METHODS = ("mean", "sine")


def add_seg_uid(df: pd.DataFrame) -> pd.DataFrame:
    """`src`·`seg`로 파일 간 유일한 세그먼트 키 `seg_uid`를 만든다(이미 있으면 그대로)."""
    if "seg_uid" in df.columns:
        return df
    if not {"src", "seg"} <= set(df.columns):
        raise KeyError("seg_uid를 만들려면 'src'와 'seg' 컬럼이 필요하다 (data_quality.load 결과를 넣을 것)")
    out = df.copy()
    out.insert(0, "seg_uid", out["src"].astype(str) + "_" + out["seg"].astype(str))
    return out


def quantize_to_step(x: np.ndarray | pd.Series, step: float = CUR_STEP) -> np.ndarray:
    """값을 step의 정수배로 반올림한다."""
    return np.round(np.asarray(x, dtype=float) / step) * step


CUR_DECIMALS = 1          # 격자 양자화 뒤 추가 반올림 자릿수 — 03 A-3: float 표기 자릿수 잔여 누수(AUC 0.657→0.581) 차단


def unify_format(df: pd.DataFrame, vib_decimals: int = VIB_DECIMALS,
                 cur_step: float | None = CUR_STEP, cur_decimals: int | None = CUR_DECIMALS) -> pd.DataFrame:
    """두 파일의 값 표기 형식을 같게 맞춘다.

    - 진동(AI0/AI1): 소수 `vib_decimals` 자리로 반올림 (정상 파일 형식 기준. 8자리 값을 6자리로 낮춤).
    - 전류(AI2): `cur_step` 격자로 양자화. 정상 파일 값은 격자 위에 있지 않으므로(0.19%) 정상 쪽을
      이상 쪽 격자에 맞추는 것이며, 격자 간격(1.19)은 전류 RMS 80~150 대비 1% 미만이라 신호 손실은 없다.
      `cur_step=None`이면 건너뛴다. 양자화 뒤 `cur_decimals` 자리로 다시 반올림한다 — k×1.19209 형태의
      float 표기는 소수 자릿수가 값 크기에 따라 달라져 형식 피처(자릿수)로 파일이 일부 구분되기 때문
      (03 A-3: 전류 형식 피처 AUC 0.657 → 1자리 반올림 후 0.581, 최대 값 변화 0.06). `None`이면 건너뛴다.

    반환은 복사본. 원본 컬럼을 덮어쓰며 새 컬럼은 만들지 않는다.
    """
    out = df.copy()
    for ch in VIB_CHANNELS:
        if ch in out.columns:
            out[ch] = out[ch].round(vib_decimals)
    if CUR_CHANNEL in out.columns:
        if cur_step is not None:
            out[CUR_CHANNEL] = quantize_to_step(out[CUR_CHANNEL].to_numpy(), cur_step)
        if cur_decimals is not None:
            out[CUR_CHANNEL] = out[CUR_CHANNEL].round(cur_decimals)
    return out


def remove_dc(df: pd.DataFrame, by: str = "seg_uid", method: str = "mean",
              channels: list[str] | None = None, keep_dc: bool = False) -> pd.DataFrame:
    """세그먼트(`by`)별 DC 오프셋을 제거한다.

    - method="mean": 세그먼트 평균을 뺀다 (02 §9 채택안).
    - method="sine": 세그먼트별 사인 피팅(A·sin(2πft+φ)+C)의 잔차를 쓴다 — 03 노트북 A-1(전류 DC가
      에일리어싱 산물인지) 게이트 G1-a일 때 구현한다. 현재는 자리만 있다.
    - keep_dc=True면 제거한 오프셋을 `<채널>_dc` 컬럼(세그먼트 상수)으로 남긴다. |DC| 자체는 형식 누수
      채널이므로 모델 입력에는 넣지 않고 진단·리포트용으로만 쓴다.
    """
    if method not in DC_METHODS:
        raise ValueError(f"method는 {DC_METHODS} 중 하나: {method!r}")
    if method == "sine":
        raise NotImplementedError("사인 피팅 잔차 방식은 03 노트북 A-1 게이트 결과(G1-a)에 따라 추가한다")
    channels = channels or [c for c in SENSORS if c in df.columns]
    out = add_seg_uid(df) if by == "seg_uid" else df.copy()
    for ch in channels:
        dc = out.groupby(by)[ch].transform("mean")
        if keep_dc:
            out[f"{ch}_dc"] = dc
        out[ch] = out[ch] - dc
    return out


def preprocess(df: pd.DataFrame, dc_method: str = "mean", **unify_kwargs) -> pd.DataFrame:
    """표준 전처리 파이프라인: add_seg_uid → unify_format → remove_dc. 모델 노트북은 이 함수만 호출한다."""
    return remove_dc(unify_format(add_seg_uid(df), **unify_kwargs), method=dc_method)


# --- 형식 피처 (진단 전용) --------------------------------------------------

def _decimal_places(x: pd.Series) -> pd.Series:
    """각 값의 소수점 이하 자릿수 (문자열 표기 기준 — signal_checks.decimal_places_table과 동일 정의)."""
    s = x.astype(str).str.split(".").str[1]
    return s.str.len().fillna(0).astype(int)


def format_features(df: pd.DataFrame, by: str = "seg_uid", cur_step: float = CUR_STEP,
                    tol: float = CUR_STEP_TOL) -> pd.DataFrame:
    """세그먼트별 **형식 피처만** 뽑는다 (물리 신호 피처 없음).

    채널별로: `<ch>_ndec_mode`(소수 자릿수 최빈값), `<ch>_ndec_mean`, `<ch>_dc_abs`(|세그먼트 평균|),
    `<ch>_uniq_ratio`(고유값 수/길이). 전류는 추가로 `AI2_Current_step_share`(격자 정수배 비율).
    반환 index는 `by`. 전처리 전/후에 각각 계산해 로지스틱 회귀 AUC를 비교하는 데 쓴다(A-3).
    """
    d = add_seg_uid(df) if by == "seg_uid" else df
    rows = []
    for key, g in d.groupby(by, sort=False):
        row = {by: key}
        for ch in SENSORS:
            if ch not in g.columns:
                continue
            x = g[ch]
            nd = _decimal_places(x)
            row[f"{ch}_ndec_mode"] = int(nd.mode().iloc[0]) if len(nd) else 0
            row[f"{ch}_ndec_mean"] = float(nd.mean())
            # DC 제거 후 남는 1e-17 수준 부동소수 잔차가 가짜 분리력을 만들지 않도록 9자리에서 자른다
            row[f"{ch}_dc_abs"] = round(float(abs(x.mean())), 9)
            row[f"{ch}_uniq_ratio"] = float(x.nunique() / len(x))
        if CUR_CHANNEL in g.columns:
            q = g[CUR_CHANNEL].to_numpy(dtype=float) / cur_step
            row[f"{CUR_CHANNEL}_step_share"] = float((np.abs(q - np.round(q)) < tol).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


FORMAT_FEATURE_SUFFIXES = ("_ndec_mode", "_ndec_mean", "_dc_abs", "_uniq_ratio", "_step_share")


def format_feature_columns(cols, channels: list[str] | None = None) -> list[str]:
    """`format_features` 결과에서 특정 채널(들)의 형식 피처 컬럼만 고른다 (채널별 차단 실험용)."""
    channels = channels or SENSORS
    return [c for c in cols if any(c.startswith(ch) and c.endswith(FORMAT_FEATURE_SUFFIXES) for ch in channels)]
