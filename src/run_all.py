"""노트북 전체 순차 실행 스크립트(재현성 점검용).

사용: (저장소 루트에서) python src/run_all.py [--from NB] [--to NB] [--only NB,NB] [--log PATH] [--dry-run]

- 실행 순서는 ORDER에 고정한다. 입력 계약(data/processed 파일, results/*/cv_scores.csv)의 의존 방향을 따른다.
- SKIP의 노트북은 돌리지 않고 사유를 로그에 남긴다(34 챗봇: 약 3GB LLM 다운로드 필요).
- 각 노트북은 notebooks/ 안에서 nbconvert --execute --inplace 로 실행하고, 소요 시간·성공 여부·오류 요약을 로그 CSV에 한 줄씩 기록한다.
- 경로는 paths.py와 같은 기준(ROOT = 이 파일의 부모의 부모)을 쓴다.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# 실행 순서(의존 방향): 진단 → 전처리·피처 → 규칙·모델 → 결합 → 분석
ORDER = [
    "01_data_quality",
    "02_diagnosis_deep",
    "03_diagnosis_recheck_CHS",
    "11_window_features_CHS",
    "04_data_profile_CHS",                     # 11의 11_window_features.parquet 입력(README 순서 01~04와 다름)
    "12_feature_statistics_CHS",
    "13_preprocess_dc_check_CHS",
    "14_gap_interpolation_check_JIW",
    "21_model_rule_baseline_CHS",
    "24_model_forecasting_JIW",
    "25_model_graph_JIW",
    "26_model_distribution_JIW",
    "27_model_cnn_lstm_ae_CHS",
    "22_model_classical_timeseries_JSC",       # 24·26 cv_scores 참조
    "34_realtime_input_validation_JSC",        # 22 cv_scores 참조
    "23_model_ensemble_ridge_mcd_JSC",         # 34_realtime cv_scores 참조
    "29_model_guidebook_lstm_ae_JIW",          # G1은 results/g1_runs.csv(저장값) 사용, G2 학습
    "28_model_ensemble_JIW",                   # 24·26 가중치·점수
    "31_error_analysis_CHS",                   # 2x metrics.csv 전부
    "32_operating_point_CHS",                  # 31 f1_matched_fpr.csv 정합 assert
    "32_alarm_explain_JIW",
    "33_early_warning_JIW",
    "35_eval_detail_JIW",
    "36_conformal_pvalue_CHS",                 # 23 구조(main에 없으면 건너뜀)
    "37_ridge_mcd_dashboard_JSC",              # 23 점수·34 처리시간·segments.csv로 정적 HTML 생성
]
SKIP = {
    "34_alarm_chatbot_JIW": "로컬 LLM(약 3GB) 다운로드가 필요해 순차 실행에서 제외. 판정 성능과 무관",
}


def run_one(nb: str, kernel: str, timeout: int) -> tuple[bool, float, str]:
    path = NB_DIR / f"{nb}.ipynb"
    if not path.exists():
        return False, 0.0, "파일 없음"
    env = dict(os.environ, PYTHONUTF8="1")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           f"--ExecutePreprocessor.kernel_name={kernel}", f"--ExecutePreprocessor.timeout={timeout}", path.name]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=NB_DIR, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if r.returncode == 0:
        return True, dt, ""
    tail = (r.stderr or r.stdout).strip().splitlines()
    msg = " | ".join(l.strip() for l in tail[-6:])[:600]
    return False, dt, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--only")
    ap.add_argument("--kernel", default="kamp")
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="노트북당 셀 타임아웃(초)")
    ap.add_argument("--log", default=str(ROOT / "run_all_log.csv"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args()

    todo = list(ORDER)
    if a.only:
        todo = [n for n in ORDER if n in set(a.only.split(","))]
    else:
        if a.start:
            todo = todo[todo.index(a.start):]
        if a.end:
            todo = todo[: todo.index(a.end) + 1]

    log = Path(a.log)
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["order", "notebook", "status", "elapsed_s", "started_at", "message"])
        for nb, why in SKIP.items():
            if not a.only:
                w.writerow(["-", nb, "skipped", 0, datetime.now().isoformat(timespec="seconds"), why])
        f.flush()
        for i, nb in enumerate(todo, 1):
            started = datetime.now().isoformat(timespec="seconds")
            print(f"[{i}/{len(todo)}] {nb} ... ", end="", flush=True)
            if a.dry_run:
                print("dry-run")
                continue
            ok, dt, msg = run_one(nb, a.kernel, a.timeout)
            status = "ok" if ok else ("missing" if msg == "파일 없음" else "error")
            print(f"{status} {dt/60:.1f}분" + (f" — {msg[:120]}" if msg else ""), flush=True)
            w.writerow([i, nb, status, round(dt, 1), started, msg])
            f.flush()
            if not ok and status == "error" and a.stop_on_error:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
