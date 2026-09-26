"""주제 ③ 프레스 유압펌프 진동·전류 시계열 — 로딩 및 품질 진단 유틸리티."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"

FILES = {"normal": "press_data_normal.csv", "outlier": "outlier_data.csv"}
SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
LABEL = "Equipment_state"
SAMPLE_SEC = 0.1          # 10 Hz (TimeStamp 간격의 최빈값)
GAP_SEC = 0.5             # 이보다 큰 간격이면 새 세그먼트(수집 burst)로 본다


def detect_encoding(path: Path, n_bytes: int = 4096) -> str:
    b = path.read_bytes()[:n_bytes]
    if b.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        b.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"


def _ensure_raw_files() -> None:
    """data/raw/에 필요한 원본 CSV가 없으면 zip에서 자동 추출한다.

    순환 임포트를 피하려고 extract 모듈은 실제로 필요할 때(파일이 없을 때)만 함수 내부에서 import한다.
    """
    missing = [name for name in FILES.values() if not (RAW_DIR / name).exists()]
    if missing:
        from extract import extract
        extract()


def file_inventory() -> pd.DataFrame:
    _ensure_raw_files()
    rows = []
    for key, name in FILES.items():
        p = RAW_DIR / name
        rows.append({"key": key, "file": name, "size_MB": round(p.stat().st_size / 1e6, 3),
                     "encoding": detect_encoding(p)})
    return pd.DataFrame(rows)


def load(key: str) -> pd.DataFrame:
    """첫 컬럼(이름 없음)은 원본 행 인덱스. TimeStamp를 datetime으로 파싱하고 세그먼트 id를 붙인다."""
    _ensure_raw_files()
    p = RAW_DIR / FILES[key]
    d = pd.read_csv(p, index_col=0, encoding=detect_encoding(p))
    d["ts"] = pd.to_datetime(d["TimeStamp"])
    d["dt"] = d["ts"].diff().dt.total_seconds()
    d["seg"] = (d["dt"] > GAP_SEC).cumsum()
    d["src"] = key
    return d


def load_all() -> dict[str, pd.DataFrame]:
    return {k: load(k) for k in FILES}


def segment_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("seg").agg(n=("ts", "size"), start=("ts", "min"), end=("ts", "max"),
                             label=(LABEL, "mean"),
                             v0_sd=("AI0_Vibration", "std"), v1_sd=("AI1_Vibration", "std"),
                             i_sd=("AI2_Current", "std"),
                             v0_absmax=("AI0_Vibration", lambda x: x.abs().max()))
    g["dur_s"] = (g["end"] - g["start"]).dt.total_seconds().round(1)
    return g


def rolling_rms(d: pd.DataFrame, win: int = 10) -> pd.DataFrame:
    """세그먼트 경계를 넘지 않는 이동 RMS (기본 1초 = 10샘플)."""
    out = d.groupby("seg")[SENSORS].rolling(win).apply(lambda x: np.sqrt(np.mean(x ** 2)), raw=True)
    return out.reset_index(level=0, drop=True).sort_index()


def zero_crossing_period(x: np.ndarray) -> float:
    """부호 변화로 추정한 평균 주기(샘플 수). AC 전류의 겉보기 주파수 확인용."""
    s = np.sign(x)
    zc = np.sum(s[1:] != s[:-1])
    return 2 * len(x) / zc if zc else np.inf


def outlier_table(d: pd.DataFrame, ref: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """ref(정상) 기준 평균/표준편차로 z를 계산해 |z|>z 비율을 센다."""
    mu, sd = ref[SENSORS].mean(), ref[SENSORS].std()
    zz = (d[SENSORS] - mu) / sd
    return pd.DataFrame({"share_abs_z_gt": (zz.abs() > z).mean().round(4),
                         "min": d[SENSORS].min().round(3), "max": d[SENSORS].max().round(3)})
