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

DC 제거 방식 (`remove_dc(method=...)`, 상세는 노트북 13 `13_preprocess_dc_check_CHS`에 둔다)
- 정상 전류의 참 DC는 거의 상수다: 길이 50 세그먼트 202개의 세그먼트 평균 q05/q50/q95 = -0.12 / 0.57 / 1.19.
  진동 두 채널은 0 근처다.
- 전류 0.6 Hz(60 Hz 에일리어싱) 한 주기는 17샘플이다. 그보다 짧은 세그먼트의 "세그먼트 평균"은 DC가 아니라
  사인 위상값이어서 평균을 빼면 AC를 깎는다. |세그먼트 평균|과 길이의 Spearman -0.835, 길이 8 이하 정상
  세그먼트 |평균| q95 182, 9~16은 90, 50은 1.2. 이론상 평균 제거 후 남는 AC RMS는 n=5에서 47%,
  n=10에서 84%(위상 평균)다.
- 이상 세그먼트의 오프셋은 상수가 아니라 느린 드리프트 성분을 포함한다(세그먼트 내 선형 추세 R² 중앙값 0.14,
  최대 0.70). 평균 제거는 이 드리프트를 남긴다.
- 하이패스/밴드패스 필터는 기각했다: DC를 5초 안에 지우는 차단 주파수(0.2 Hz 이하)는 0.6 Hz AC를 40~55%
  왜곡하고, causal 필터는 세그먼트 시작 1초 과도 오차가 AC 표준편차(82)를 넘는다.
- 사인 피팅 DC는 03 A-1(G1-b) 결과 채택하지 않았다(1주기 미만 세그먼트에서 피팅이 불안정).
- `mean`: 세그먼트 평균 제거. 오프라인 benchmark 전처리(benchmark_v1)로 유지한다.
- `baseline`: 학습 정상 긴 세그먼트에서 추정한 채널별 상수(`fit_baseline`)를 뺀다. 길이 무관·causal한 후보.
  **단, 이상 파일의 세그먼트 DC 오프셋(02 §2 형식 누수)은 제거되지 않는다** — 상수를 빼도 세그먼트 간 순위는
  그대로라 |세그먼트 평균| 단독 AUC가 AI0 0.9443 / AI1 0.9198 / 전류 0.8911로 `mean` 적용 전과 같다. 03 A-3의
  누수 차단 결과는 `mean` 기준이며, `baseline` 입력을 쓰는 모델은 이상의 DC를 "신호"로 보게 되므로 판정 보드 #5
  (실제 이상 vs 계측 체인 변경)의 한계를 함께 적어야 한다. `expanding`도 진동 채널 누수(0.90 / 0.86)가 대부분 남는다.
- `expanding`·`detrend`: ablation용. 각각 세그먼트 내 누적 평균(causal), 세그먼트 1차 선형 추세를 뺀다.
- 센서 열에 NaN이 있으면 `expanding`(누적 개수에 NaN 포함)·`detrend`(세그먼트 전체 NaN)는 `mean`과 달리 NaN을
  건너뛰지 않는다. 현재 데이터는 NaN 0건이라 영향이 없고, 결측이 생기면 호출 전에 처리한다.
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

DC_METHODS = ("mean", "baseline", "expanding", "detrend")
BASELINE_MIN_LEN = 17     # 0.6 Hz 한 주기(10 Hz x 1/0.6) 이상 길이의 세그먼트만 기저선 추정에 사용


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


def fit_baseline(df_train: pd.DataFrame, by: str = "seg_uid", channels: list[str] | None = None,
                 min_len: int = BASELINE_MIN_LEN, stat: str = "median") -> dict[str, float]:
    """학습 정상 데이터에서 채널별 DC 기저선(상수)을 추정한다 — `remove_dc(method="baseline")`용.

    - `df_train`은 **학습 fold의 정상 샘플만** 담은 프레임이어야 한다(형식 통일 이후, DC 제거 전).
      라벨 열이 있어도 여기서 거르지 않는다 — 정상만 넘기는 것은 호출자 책임이다.
    - 길이 `min_len`(기본 17 = 0.6 Hz 한 주기) 이상인 세그먼트의 세그먼트 평균을 모아 `stat`
      ("median" 또는 "mean")으로 집계한 값 하나를 채널별로 반환한다. 짧은 세그먼트의 평균은 DC가 아니라
      사인 위상값이므로 쓰지 않는다(모듈 docstring 참고).
    - 조건을 만족하는 세그먼트가 0개면 `ValueError`.
    - 반환 `{채널: float}`는 `remove_dc(method="baseline", baseline=...)`에 그대로 넣는다.
    """
    if stat not in ("median", "mean"):
        raise ValueError(f"stat은 'median' 또는 'mean': {stat!r}")
    channels = channels or [c for c in SENSORS if c in df_train.columns]
    d = add_seg_uid(df_train) if by == "seg_uid" else df_train
    sizes = d.groupby(by).size()
    keep = sizes.index[sizes >= min_len]
    if len(keep) == 0:
        raise ValueError(f"길이 min_len={min_len} 이상인 세그먼트가 없다 (세그먼트 수 {len(sizes)}, 조건 만족 0)")
    sub = d[d[by].isin(keep)]
    out = {}
    for ch in channels:
        seg_mean = sub.groupby(by)[ch].mean()
        out[ch] = float(seg_mean.median() if stat == "median" else seg_mean.mean())
    return out


def remove_dc(df: pd.DataFrame, by: str = "seg_uid", method: str = "mean",
              channels: list[str] | None = None, keep_dc: bool = False,
              baseline: dict[str, float] | None = None) -> pd.DataFrame:
    """세그먼트(`by`)별 DC 오프셋을 제거한다. 반환은 복사본이며 원본 채널 열을 덮어쓴다.

    - method="mean": 세그먼트 평균(상수)을 뺀다 (02 §9 채택안, 오프라인 benchmark 전처리). 1주기(17샘플)
      미만 세그먼트에서는 위상값을 빼 AC를 깎는다.
    - method="baseline": `baseline[채널]` 상수를 뺀다(`fit_baseline` 결과). 길이 무관·causal. `baseline`이
      None이거나 채널 키가 빠지면 `ValueError`. 이상 파일의 세그먼트 DC(형식 누수)는 남는다(모듈 docstring 참고).
    - method="expanding": 세그먼트 안에서 첫 샘플부터 현재 샘플까지의 누적 평균(causal)을 뺀다. 첫 샘플은
      항상 0이 된다. 1주기 미만 구간에서는 `mean`과 같은 위상 편향을 가지며 ablation용이다.
    - method="detrend": 세그먼트별 1차 선형 추세(샘플 순서 기준, `np.polyfit(deg=1)`)를 뺀다. 느린 드리프트까지
      제거하는 ablation용이다. 길이 3 미만 세그먼트는 평균 제거로 대체한다.
    - 세그먼트 내부 행 순서가 시간순이라고 가정하며, 행 순서는 바꾸지 않는다.
    - keep_dc=True면 제거한 오프셋을 `<채널>_dc` 컬럼으로 남긴다(mean=세그먼트 상수, baseline=기저선 상수,
      expanding=샘플별 누적 평균, detrend=샘플별 추세값). |DC| 자체는 형식 누수 채널이므로 모델 입력에는
      넣지 않고 진단·리포트용으로만 쓴다.
    """
    if method not in DC_METHODS:
        raise ValueError(f"method는 {DC_METHODS} 중 하나: {method!r}")
    channels = channels or [c for c in SENSORS if c in df.columns]
    if method == "baseline":
        missing = [ch for ch in channels if baseline is None or ch not in baseline]
        if missing:
            raise ValueError(f"method='baseline'에는 채널 {missing}의 baseline 값이 필요하다 (fit_baseline 결과를 넣을 것)")
    # add_seg_uid는 seg_uid가 이미 있으면 같은 객체를 돌려주므로, 입력 프레임이 바뀌지 않게 항상 복사한다
    out = add_seg_uid(df) if by == "seg_uid" else df
    if out is df:
        out = df.copy()
    for ch in channels:
        if method == "mean":
            dc = out.groupby(by)[ch].transform("mean")
        elif method == "baseline":
            dc = pd.Series(float(baseline[ch]), index=out.index)
        elif method == "expanding":
            # 위치 기반 누적합 / 누적 개수 — 인덱스가 중복(concat)이어도 행 정렬이 유지된다
            g = out.groupby(by)[ch]
            dc = g.cumsum() / (g.cumcount() + 1)
        else:  # detrend
            x = out[ch].to_numpy(dtype=float)
            trend = np.empty_like(x)
            for pos in out.groupby(by).indices.values():
                n = len(pos)
                if n < 3:
                    trend[pos] = x[pos].mean()
                else:
                    t = np.arange(n, dtype=float)
                    slope, icpt = np.polyfit(t, x[pos], 1)
                    trend[pos] = slope * t + icpt
            dc = pd.Series(trend, index=out.index)
        if keep_dc:
            out[f"{ch}_dc"] = dc
        out[ch] = out[ch] - dc
    return out


def preprocess(df: pd.DataFrame, dc_method: str = "mean", baseline: dict[str, float] | None = None,
               **unify_kwargs) -> pd.DataFrame:
    """표준 전처리 파이프라인: add_seg_uid → unify_format → remove_dc. 모델 노트북은 이 함수만 호출한다.

    기본 호출(`preprocess(df)`)은 `mean` 방식(benchmark_v1)이다. `dc_method="baseline"`이면 `fit_baseline`
    결과를 `baseline`으로 넘긴다.
    """
    return remove_dc(unify_format(add_seg_uid(df), **unify_kwargs), method=dc_method, baseline=baseline)


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
