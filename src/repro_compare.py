"""재실행 결과(작업 트리)와 커밋본의 results/**/*.csv 대조 — 재현성 점검용.

사용: (저장소 루트에서) python src/repro_compare.py [--ref REF] [--map PREFIX=REF ...] [--tol 1e-9] [--out PATH]

- 작업 트리에서 변경된(또는 새로 생긴) results/ 아래 CSV마다 커밋본(기본 HEAD)을 읽어 shape, 공통 수치열의 최대 절대차,
  tol 초과 셀 수, 비수치열 불일치 셀 수를 출력한다. `--map results/31_error_analysis_CHS/=origin/fix/...` 처럼
  경로 접두사별로 다른 참조를 줄 수 있다(PR 대기 중인 노트북).
- 소요 시간 열(`sec`, `*_sec`, `timing`)은 환경마다 달라 비교에서 제외한다.
"""
from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIME_COLS = {"sec", "fit_sec", "score_sec", "elapsed_s"}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def read_ref(rel: str, ref: str) -> pd.DataFrame | None:
    r = git("show", f"{ref}:{rel}")
    if r.returncode:
        return None
    return pd.read_csv(io.BytesIO(r.stdout), encoding="utf-8-sig")


def compare(rel: str, ref: str, tol: float) -> dict:
    new = pd.read_csv(ROOT / rel, encoding="utf-8-sig")
    old = read_ref(rel, ref)
    rec = dict(file=rel, ref=ref, status="", rows_ref=np.nan, rows_new=len(new), max_abs_diff=np.nan, n_over_tol=np.nan, n_text_diff=np.nan)
    if old is None:
        rec["status"] = "new file"
        return rec
    rec["rows_ref"] = len(old)
    if old.shape != new.shape or list(old.columns) != list(new.columns):
        rec["status"] = "SHAPE"
        return rec
    num = [c for c in old.columns if c not in TIME_COLS
           and pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c])]
    txt = [c for c in old.columns if c not in num and c not in TIME_COLS]
    over = 0
    if num:
        a, b = old[num].to_numpy(float), new[num].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)                       # 한쪽만 NaN이면 불일치
        rec["max_abs_diff"] = float(d.max()) if d.size else 0.0
        over = int((d > tol).sum())
    rec["n_over_tol"] = over
    rec["n_text_diff"] = int((old[txt].fillna("").astype(str) != new[txt].fillna("").astype(str)).to_numpy().sum()) if txt else 0
    rec["status"] = "identical" if over == 0 and rec["n_text_diff"] == 0 else "DIFF"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--map", nargs="*", default=[], help="PREFIX=REF (경로 접두사별 참조)")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", default=str(ROOT / "repro_compare.csv"))
    a = ap.parse_args()
    maps = [m.split("=", 1) for m in a.map]

    def ref_for(rel: str) -> str:
        for pre, ref in maps:
            if rel.startswith(pre):
                return ref
        return a.ref

    changed = git("diff", "--name-only", "--", "results").stdout.decode().split()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "results").stdout.decode().split()
    rows = [compare(rel, ref_for(rel), a.tol) for rel in sorted(set(changed + untracked)) if rel.endswith(".csv")]
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_rows", 1000, "display.max_colwidth", 80)
    print(out.to_string(index=False) if len(out) else "변경된 results CSV 없음")
    if len(out):
        print("\nsummary:", out.status.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
