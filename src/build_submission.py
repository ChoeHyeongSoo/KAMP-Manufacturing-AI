"""대회 제출용 소스코드 폴더·zip 생성 스크립트.

사용: (저장소 루트에서) python src/build_submission.py [--out submission] [--name kamp_press_pump]
                                                      [--models final|all|none] [--no-zip]

- 허용 목록(allowlist) 방식으로만 복사한다. 목록에 없는 것은 복사하지 않는다.
- 의도적으로 제외하는 것(목록에 없으므로 자동 제외):
  .git · .venv · .claude · .github · .gitignore · .gitattributes · CLAUDE.md  : 개발 도구·팀 작업 규칙(블라인드 평가 식별 정보 우려)
  topics/                         : 선정하지 않은 다른 주제의 자료
  data/ 의 다른 zip 4개            : 주제 ③ 외 데이터(용량·불필요)
  data/processed/                 : 노트북이 재생성하는 중간 산출물
  notebooks/scratch/              : 개인 실험
  docs/ 의 나머지                  : 팀 내부 설계·회의 문서(재현성 점검 문서만 포함)
  run_all_log.csv, submission/    : 로컬 실행 로그, 출력물 자체
- 복사 후 `CLAUDE.md` 언급을 중립 문구로 치환하고, 금지 문자열(도구명·개인 식별 정보·절대경로)을 검사한다.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL_MODELS = [
    "13_preprocess_dc_check_CHS",
    "22_model_classical_timeseries_JSC",
    "23_model_ensemble_ridge_mcd_JSC",
    "34_realtime_input_validation_JSC",
]
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", "scratch", "_template"}
SINGLE_FILES = [  # (원본, 출력 내 경로)
    ("requirements.txt", "requirements.txt"),
    ("data/README.md", "data/README.md"),
    ("results/README.md", "results/README.md"),
    ("data/raw/press_data_normal.csv", "data/raw/press_data_normal.csv"),
    ("data/raw/outlier_data.csv", "data/raw/outlier_data.csv"),
    ("data/3. 소성가공 예지보전 AI 데이터셋.zip", "data/3. 소성가공 예지보전 AI 데이터셋.zip"),
    ("docs/reproducibility_check.md", "docs/reproducibility_check.md"),
    ("docs/submission_README.md", "README.md"),
]
REPLACE_FILES = ["notebooks/31_error_analysis_CHS.ipynb", "reports/31_error_analysis_CHS.md"]
TEXT_EXT = {".py", ".ipynb", ".md", ".txt", ".csv", ".html", ".json"}
FORBIDDEN = [
    ("claude", r"claude"), ("anthropic", r"anthropic"), ("copilot", r"copilot"), ("chatgpt", r"chatgpt"),
    ("CLAUDE.md", r"CLAUDE\.md"), ("C:/Users", r"C:[\/]+Users"), ("/Users/", r"/Users/"),
    ("ChoeHyeongSoo", r"ChoeHyeongSoo"), ("inwoonghub", r"inwoonghub"), ("JisuChung", r"JisuChung"),
    ("github.com/", r"github\.com/"), ("email", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]
FORBIDDEN_RE = [(n, re.compile(p, re.I)) for n, p in FORBIDDEN]


def copy_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.suffix == ".pyc":
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def copy_glob(pattern_dir: str, glob: str, dst: Path, skip: set[str] = frozenset()) -> None:
    for p in sorted((ROOT / pattern_dir).glob(glob)):
        if p.is_file() and p.name not in skip:
            out = dst / pattern_dir / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def scan_text(text: str, label: str, hits: list) -> None:
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in FORBIDDEN_RE:
            if rx.search(line):
                hits.append((label, i, name, line.strip()[:120]))


def scan_notebook(path: Path, rel: str, hits: list) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for ci, cell in enumerate(nb.get("cells", []), 1):
        src = cell.get("source", [])
        scan_text("".join(src) if isinstance(src, list) else src, f"{rel}#cell{ci}.source", hits)
        for out in cell.get("outputs", []):
            # 이미지(base64)는 제외하고 텍스트 계열 출력 전부(stdout·stderr·traceback·text/*)를 검사한다
            if out.get("output_type") == "stream":
                t = out.get("text", "")
                scan_text("".join(t) if isinstance(t, list) else t, f"{rel}#cell{ci}.{out.get('name', 'stream')}", hits)
            if out.get("output_type") == "error":
                scan_text("\n".join(out.get("traceback", [])), f"{rel}#cell{ci}.traceback", hits)
            for mime, val in out.get("data", {}).items():
                if mime.startswith("text/"):
                    scan_text("".join(val) if isinstance(val, list) else str(val), f"{rel}#cell{ci}.{mime}", hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission")
    ap.add_argument("--name", default="kamp_press_pump")
    ap.add_argument("--models", choices=["final", "all", "none"], default="final")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    out_root = (ROOT / a.out)
    dst = out_root / a.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # 코드·노트북
    # 이 스크립트 자체는 검사용 금지 문자열 목록을 담고 있어 제출물에서 뺀다
    copy_glob("src", "*.py", dst, skip={Path(__file__).name})
    copy_glob("notebooks", "*.ipynb", dst)
    # 산출물(하위 폴더 전부, _template 제외)
    for d in ["figures", "results", "reports", "predictions"]:
        if (ROOT / d).is_dir():
            copy_tree(ROOT / d, dst / d)
        else:
            print(f"[건너뜀] {d}/ 없음")
    # 단일 파일
    for s, t in SINGLE_FILES:
        sp = ROOT / s
        if not sp.exists():
            print(f"[경고] 없음: {s}")
            continue
        (dst / t).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dst / t)
    # 모델 가중치
    if a.models == "all":
        copy_tree(ROOT / "models", dst / "models")
    elif a.models == "final":
        for m in FINAL_MODELS:
            if (ROOT / "models" / m).is_dir():
                copy_tree(ROOT / "models" / m, dst / "models" / m)
            else:
                print(f"[경고] 없음: models/{m}")

    # 텍스트 치환
    n_sub = 0
    for rel in REPLACE_FILES:
        p = dst / rel
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        c = t.count("CLAUDE.md")
        t = t.replace("`CLAUDE.md`", "팀 규칙 문서").replace("CLAUDE.md", "팀 규칙 문서")
        p.write_text(t, encoding="utf-8", newline="")
        n_sub += c
        print(f"치환 {rel}: {c}건")

    # 금지 문자열 검사
    hits: list = []
    for p in sorted(dst.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        rel = p.relative_to(dst).as_posix()
        try:
            if p.suffix.lower() == ".ipynb":
                scan_notebook(p, rel, hits)
            else:
                scan_text(p.read_text(encoding="utf-8", errors="replace"), rel, hits)
        except Exception as e:  # noqa: BLE001
            hits.append((rel, 0, "parse-error", str(e)))

    files = [p for p in dst.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    zip_size = None
    if not a.no_zip:
        zp = out_root / f"{a.name}.zip"
        if zp.exists():
            zp.unlink()
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):
                z.write(p, f"{a.name}/{p.relative_to(dst).as_posix()}")
        zip_size = zp.stat().st_size

    print(f"파일 수: {len(files)}  총 용량: {size/1e6:.1f} MB"
          + (f"  zip: {zip_size/1e6:.1f} MB" if zip_size is not None else "  (zip 생략)"))
    print(f"치환 건수: {n_sub}")
    if hits:
        print(f"금지 문자열 검사: 실패 {len(hits)}건")
        for h in hits[:200]:
            print(f"  {h[0]}:{h[1]} [{h[2]}] {h[3]}")
        return 1
    print("금지 문자열 검사: 통과(0건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
