"""공용 경로 상수 및 노트북별 출력 폴더 헬퍼.

3명이 동시에 노트북을 작업할 때 `figures/`, `results/` 아래 파일명이 겹치지 않도록,
노트북 하나당 `figures/<노트북명>/`, `results/<노트북명>/` 서브폴더를 쓰기로 한다.
`nb_dirs()`로 그 두 폴더를 얻고(없으면 생성), `save_fig()`로 그림을 저장한다.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def nb_dirs(nb_name: str) -> tuple[Path, Path]:
    """노트북 파일명(확장자 제외, 예: "21_model_iforest_CHS")별 그림/결과 폴더를 만들어 반환한다.

    반환: (figures/<nb_name>/, results/<nb_name>/). 둘 다 없으면 생성한다(mkdir parents, exist_ok).
    """
    if nb_name.endswith(".ipynb"):
        nb_name = nb_name[: -len(".ipynb")]
    if "/" in nb_name or "\\" in nb_name:
        raise ValueError(f"nb_name에 경로 구분자를 쓸 수 없음: {nb_name!r}")

    fig_dir = FIGURES / nb_name
    res_dir = RESULTS / nb_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


def save_fig(fig_dir: Path, name: str, dpi: int = 110) -> None:
    """현재 matplotlib figure를 fig_dir/name으로 저장하고 닫는다."""
    import matplotlib.pyplot as plt

    plt.tight_layout()
    path = fig_dir / name
    plt.savefig(path, dpi=dpi)
    plt.close()
    print("saved", path.relative_to(ROOT))
