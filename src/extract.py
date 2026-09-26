"""프레스 유압펌프 원본 zip -> data/raw/ 해제.

`data/3. 소성가공 예지보전 AI 데이터셋.zip` 안에는 `3. 소성가공 예지보전 AI 데이터셋/` 폴더 하나에
CSV 2개(press_data_normal.csv, outlier_data.csv)가 들어 있다. 이 폴더 계층을 없애고
data/raw/ 바로 아래 평탄화해서 푼다. 이미 두 파일이 모두 있으면 건너뛰므로(--force 없이) 여러 번
실행해도 안전하다(idempotent).

실행: python src/extract.py [--force]
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "3. 소성가공 예지보전 AI 데이터셋.zip"
RAW_DIR = ROOT / "data" / "raw"
TARGET_FILES = ("press_data_normal.csv", "outlier_data.csv")


def fix_name(info: zipfile.ZipInfo) -> str:
    """zip 내부 한글 파일명 복원.

    UTF-8 플래그(0x800)가 설정돼 있으면 zipfile이 이미 올바르게 디코딩한 것이므로 그대로 쓰고,
    아니면 cp437로 잘못 해석된 바이트를 cp949로 다시 디코딩한다(다른 주제 zip 대응).
    """
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def extract(force: bool = False) -> list[str]:
    """zip에서 TARGET_FILES 2개를 data/raw/에 평탄화해서 꺼낸다. 이미 있으면 건너뛴다(idempotent).

    반환: 이번 호출에서 실제로 새로 추출한 파일명 목록(건너뛴 경우 빈 리스트).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = {name: RAW_DIR / name for name in TARGET_FILES}

    if not force and all(p.exists() for p in targets.values()):
        print(f"이미 존재, 건너뜀: {', '.join(targets)} (재추출하려면 --force)")
        return []

    if not ZIP_PATH.exists():
        sys.exit(f"zip 없음: {ZIP_PATH}\n→ data/README.md 안내대로 KAMP 포털에서 받아 data/에 두세요.")

    written: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            base = Path(fix_name(info)).name
            dst = targets.get(base)
            if dst is None:
                continue
            if dst.exists() and not force:
                continue
            with z.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            written.append(base)
            print(f"추출: {base}")

    missing = [name for name, p in targets.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"zip 안에서 못 찾음: {missing}")
    return written


if __name__ == "__main__":
    extract(force="--force" in sys.argv)
