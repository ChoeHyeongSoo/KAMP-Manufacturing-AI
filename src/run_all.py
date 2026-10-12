"""노트북 전체 순차 실행 스크립트(재현성 점검용).

사용: (저장소 루트에서) python src/run_all.py [--quick] [--from NB] [--to NB] [--only NB,NB] [--log PATH] [--dry-run]
                                              [--install] [--skip-check] [--kernel NAME]

- 시작할 때 필수 패키지(import·requirements.txt 고정 버전)를 점검한다. 없으면 설치 명령을 안내하고 종료 코드 2로 멈춘다(--install 이면 설치까지 수행).
- --quick 은 결정적 노트북 7개(01·02·03·11·22·34·23, 재학습 없음, CPU 약 3분)만 돌린다. 전체는 약 2.5시간.
- 끝나면 최종 모델(23)의 핵심 수치를 results/summary_final.csv 로 모아 표준 출력에 찍는다.
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
# --quick: 결정적 노트북만(재학습 없음, CPU 약 3분). 최종 모델 23까지의 입력 계약 순서
QUICK = [
    "01_data_quality", "02_diagnosis_deep", "03_diagnosis_recheck_CHS", "11_window_features_CHS",
    "22_model_classical_timeseries_JSC", "34_realtime_input_validation_JSC", "23_model_ensemble_ridge_mcd_JSC",
]
# 의존성 점검: (import 이름, requirements.txt 패키지명). 선택 패키지(챗봇 34)는 경고만
REQUIRED = [("pandas", "pandas"), ("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
            ("matplotlib", "matplotlib"), ("seaborn", "seaborn"), ("pyarrow", "pyarrow"), ("torch", "torch"),
            ("pyod", "pyod"), ("tqdm", "tqdm"), ("nbconvert", "nbconvert"), ("nbformat", "nbformat"),
            ("ipykernel", "ipykernel")]
OPTIONAL = [("transformers", "transformers"), ("accelerate", "accelerate")]
FINAL_NB = "23_model_ensemble_ridge_mcd_JSC"


def pinned_versions() -> dict[str, str]:
    """requirements.txt의 `pkg==ver` 행을 {pkg: ver}로 읽는다."""
    out: dict[str, str] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            pkg, ver = line.split("==", 1)
            out[pkg.strip().lower()] = ver.strip()
    return out


def check_deps(install: bool) -> bool:
    """필수 패키지 import·고정 버전 일치를 확인한다. 빠지면 설치 명령을 안내하고 False.

    --install 이면 `pip install -r requirements.txt`를 한 번 실행한 뒤 다시 확인한다.
    """
    from importlib import import_module, metadata

    pins = pinned_versions()
    missing, mismatch = [], []
    for mod, pkg in REQUIRED:
        try:
            import_module(mod)
        except Exception:  # noqa: BLE001
            missing.append(pkg)
            continue
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
        want = pins.get(pkg.lower())
        if want and not have.startswith(want):
            mismatch.append(f"{pkg} {have} (requirements {want})")
    for mod, pkg in OPTIONAL:
        try:
            import_module(mod)
        except Exception:  # noqa: BLE001
            print(f"[안내] 선택 패키지 {pkg} 없음 - 34 챗봇 노트북만 영향, 판정 성능과 무관")
    if mismatch:
        print("[경고] 고정 버전과 다른 패키지(실행은 계속): " + "; ".join(mismatch))
    if not missing:
        return True
    cmd = f"{sys.executable} -m pip install -r requirements.txt"
    if install:
        print(f"[설치] 누락 {', '.join(missing)} → {cmd}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], cwd=ROOT)
        if r.returncode == 0:
            return check_deps(install=False)
    print(f"[중단] 필수 패키지 없음: {', '.join(missing)}\n"
          f"  저장소 루트에서 다음을 실행한 뒤 다시 시도: {cmd}\n"
          f"  (torch 설치가 무거우면 CPU 빌드: {sys.executable} -m pip install torch==2.14.0 "
          f"--index-url https://download.pytorch.org/whl/cpu)\n"
          f"  또는 python src/run_all.py --install 로 설치까지 한 번에 진행")
    return False


def write_final_summary() -> None:
    """최종 모델(23)의 핵심 수치를 results/summary_final.csv로 모으고 표준 출력에 찍는다."""
    import pandas as pd

    res = ROOT / "results" / FINAL_NB
    if not (res / "summary.csv").exists() or not (res / "operating_decision.csv").exists():
        print(f"[안내] {res.relative_to(ROOT)} 결과 없음 — 최종 요약 생략")
        return
    summ = pd.read_csv(res / "summary.csv")
    dec = pd.read_csv(res / "operating_decision.csv")
    rows = []
    for _, d in dec.iterrows():
        for split in ("time_block", "group_kfold_seg"):
            s = summ[(summ["model"] == d["rule"]) & (summ["split"] == split)]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append({"stage": d["stage"], "rule": d["rule"], "split": split, "auc": round(float(s["auc"]), 4),
                         "fpr_segment": round(float(s["fpr_segment"]), 4), "fpr_sample": round(float(s["fpr_sample"]), 4),
                         "detected_segments": d["detected"] if split == "time_block" else "",
                         "delay_median_s": s["delay_median_s"]})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "summary_final.csv", index=False, encoding="utf-8")
    print("\n=== 최종 모델(Ridge-VAR + MCD 2단계 경보, 노트북 23) 요약 → results/summary_final.csv ===")
    print(out.to_string(index=False))
    print("테스트 데이터(이상 파일) 예측결과: predictions/ (python src/export_predictions.py 로 재생성)")


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
    # Windows cp949 콘솔에서도 깨지지 않게: 인코딩 불가 문자는 치환
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--only")
    ap.add_argument("--kernel", default="python3", help="nbconvert 커널 이름(ipykernel 기본 등록명). 팀 로컬 환경은 --kernel kamp")
    ap.add_argument("--timeout", type=int, default=4 * 3600, help="노트북당 셀 타임아웃(초)")
    ap.add_argument("--log", default=str(ROOT / "run_all_log.csv"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--quick", action="store_true", help="결정적 노트북 7개만(최종 모델 23 포함, CPU 약 3분)")
    ap.add_argument("--install", action="store_true", help="필수 패키지가 없으면 pip install -r requirements.txt 실행")
    ap.add_argument("--skip-check", action="store_true", help="의존성 점검 생략")
    a = ap.parse_args()

    if not a.dry_run and not a.skip_check and not check_deps(a.install):
        return 2

    todo = list(ORDER)
    if a.quick:
        todo = list(QUICK)
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
    if not a.dry_run:
        write_final_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
