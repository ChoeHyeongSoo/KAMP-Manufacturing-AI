"""주제 ④ X-ray zip → topics/04_xray/data_raw/ 용도별 해제.

원본 zip 구조가 실습용(코드·가중치·중복 이미지 세트 혼재)이라 아래처럼 나눠 푼다.
이미 풀린 파일은 건너뛰므로 여러 번 실행해도 안전하다.

data_raw/
├─ raw_bmp/machine_{1,2,3}/   원본 BMP (1호기·2호기·3호기, 2020-06-22 ~ 09-22). zip에서는
│                             `<호기>/<시리얼>_<날짜>_NgImage/` 하위 폴더에 있으나 평탄화한다.
│                             같은 파일이 인접 날짜 폴더에 중복 수록된 277건(바이트 동일)은 하나로 합쳐짐.
├─ raw_bmp/index.csv          평탄화로 사라지는 정보 보존: basename, machine, serial, folder_date,
│                             filename_date, n_copies, size, crc
├─ labeled/images/            라벨 대응 JPG 400장 (images 15⊂50⊂...⊂400 세트의 합집합)
├─ labeled/labels/            YOLO txt 500개 (class 0 = defect, cx cy w h 정규화). 500개 모두 raw_bmp에 원본 있음
├─ labeled/voc_xml/           OpenLabeling 출력 PASCAL VOC xml (참고용)
└─ reference/
   ├─ yolov3/                 참고 yolov3 코드·cfg·train/test 목록(15장)·학습 로그 (BMP 폴더 제외)
   ├─ weights/                실습별 가중치 *.pt (약 1.7 GB)
   └─ openlabeling/           라벨링 도구 소스 (input/output 제외)

실행: python topics/04_xray/src/extract.py [--force]   (프로젝트 루트 또는 어디서든)
"""
from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ZIP = ROOT / "data" / "4. X-ray 검사장비 AI 데이터셋.zip"
OUT = ROOT / "topics" / "04_xray" / "data_raw"
PREFIX = "4. X-ray 검사장비 AI 데이터셋/dataset/"
MACHINE = {"1호기": "machine_1", "2호기": "machine_2", "3호기": "machine_3"}


def fix_name(name: str) -> str:
    """zip 내부 한글 파일명(cp437로 잘못 해석됨) 복원."""
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def route(rel: str) -> Path | None:
    """zip 내부 상대경로(dataset/ 이하) → data_raw 하위 목적지. None이면 건너뜀."""
    parts = rel.split("/")
    base = parts[-1]
    if not base:
        return None

    if parts[0] == "test1" and len(parts) > 2 and parts[1] == "yolov3":
        sub = parts[2]
        if sub.startswith("X선이물검출기"):
            m = re.match(r"(\d호기)", parts[3])
            if not m:
                return None
            return OUT / "raw_bmp" / MACHINE[m.group(1)] / base
        if sub == "weights" or base.endswith(".pt"):
            return OUT / "reference" / "weights" / base
        if base.endswith(".pyc"):
            return None
        return OUT / "reference" / "yolov3" / Path(*parts[2:])

    if parts[0] == "라벨링 6종 세트":
        if parts[1] == "labels":
            return OUT / "labeled" / "labels" / base
        if parts[1].startswith("images"):
            return OUT / "labeled" / "images" / base  # 세트 간 중복은 같은 경로로 합쳐짐
        return None

    if parts[0] == "실습별 가중치파일":
        return OUT / "reference" / "weights" / base

    if parts[0] == "OpenLabeling-master":
        if "output" in parts and base.endswith(".xml"):
            return OUT / "labeled" / "voc_xml" / base
        if base.endswith(".pyc") or "/output/" in rel or "/input/" in rel:
            return None
        return OUT / "reference" / "openlabeling" / Path(*parts[1:])

    if base.endswith(".ipynb"):
        return OUT / "reference" / base
    return None


def extract(force: bool = False) -> dict[str, int]:
    if not ZIP.exists():
        sys.exit(f"zip 없음: {ZIP}\n→ data/README.md 안내대로 KAMP 포털에서 받아 data/에 두세요.")
    stats = {"written": 0, "skipped": 0, "ignored": 0}
    index: dict[tuple[str, str], dict] = {}
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = fix_name(info.filename)
            if not name.startswith(PREFIX):
                stats["ignored"] += 1
                continue
            rel = name[len(PREFIX):]
            dst = route(rel)
            if dst is None:
                stats["ignored"] += 1
                continue
            if dst.parent.parent.name == "raw_bmp":
                _index_bmp(index, rel, dst, info)
            if dst.exists() and not force and dst.stat().st_size == info.file_size:
                stats["skipped"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            stats["written"] += 1
    _write_index(index)
    return stats


def _index_bmp(index: dict, rel: str, dst: Path, info: zipfile.ZipInfo) -> None:
    """raw_bmp 평탄화로 사라지는 시리얼·폴더 날짜를 index.csv용으로 모은다."""
    folder = rel.split("/")[-2]  # 예: SN77128_20200623_NgImage
    m = re.match(r"(SN\d+)_(\d{8})_", folder)
    serial, folder_date = (m.group(1), m.group(2)) if m else ("", "")
    fm = re.match(r"\d+_(\d{8})_(\d{6})", dst.name)
    key = (dst.parent.name, dst.name)
    row = index.setdefault(key, {
        "basename": dst.name, "machine": dst.parent.name, "serial": serial,
        "folder_date": folder_date, "filename_date": fm.group(1) if fm else "",
        "filename_time": fm.group(2) if fm else "", "n_copies": 0,
        "size": info.file_size, "crc": info.CRC, "crc_conflict": 0,
    })
    row["n_copies"] += 1
    if row["folder_date"] > folder_date:  # 가장 이른 폴더 날짜를 대표로
        row["folder_date"] = folder_date
    if row["crc"] != info.CRC:
        row["crc_conflict"] = 1


def _write_index(index: dict) -> None:
    import csv

    path = OUT / "raw_bmp" / "index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(index.values(), key=lambda r: (r["machine"], r["basename"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    dup = sum(1 for r in rows if r["n_copies"] > 1)
    conflict = sum(r["crc_conflict"] for r in rows)
    print(f"index.csv: {len(rows)} unique bmp, {dup} duplicated in zip, {conflict} crc conflicts")


def summary() -> None:
    for d in sorted(p for p in OUT.rglob("*") if p.is_dir()):
        files = [f for f in d.iterdir() if f.is_file()]
        if files:
            mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"{d.relative_to(OUT)!s:45} {len(files):6d} files {mb:8.1f} MB")


if __name__ == "__main__":
    force = "--force" in sys.argv
    print(extract(force=force))
    summary()
