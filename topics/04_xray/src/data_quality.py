"""주제 ④ X-ray 데이터 — 품질 진단 유틸리티.

노트북(notebooks/01_data_quality.ipynb)에서 공통으로 사용한다.
원본 이미지·라벨은 절대 수정하지 않는다. 모든 경로는 인자로 받는 root(주제 폴더, 예:
topics/04_xray) 기준으로 계산하며 절대경로를 하드코딩하지 않는다.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MACHINES = ["machine_1", "machine_2", "machine_3"]
# 파일명 스템: <일련번호>_<YYYYMMDD>_<HHMMSS>(<연사번호>)
STEM_RE = re.compile(r"^(?P<seq>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\((?P<burst>\d+)\)$")
# BMP 팔레트에서 순수 그레이스케일(항등 매핑)이 끝나고 예약 컬러(오버레이 마커)가
# 시작되는 인덱스. raw_bmp 3개 호기 샘플 모두 244부터 비항등(RGB 서로 다름) 확인.
MARKER_MIN_INDEX = 244


def raw_bmp_dir(root: Path) -> Path:
    return Path(root) / "data_raw" / "raw_bmp"


def labeled_dir(root: Path) -> Path:
    return Path(root) / "data_raw" / "labeled"


def raw_bmp_paths(root: Path) -> list[Path]:
    return sorted((raw_bmp_dir(root)).glob("machine_*/*.bmp"))


def labeled_image_paths(root: Path) -> list[Path]:
    return sorted((labeled_dir(root) / "images").glob("*.jpg"))


def load_index(root: Path) -> pd.DataFrame:
    """raw_bmp/index.csv 로드. 날짜·시각·연사번호를 파일명에서 다시 파싱해 검증용으로 붙인다."""
    df = pd.read_csv(
        raw_bmp_dir(root) / "index.csv",
        dtype={"folder_date": str, "filename_date": str, "filename_time": str},
    )
    parsed = df["basename"].str.replace(".bmp", "", regex=False).str.extract(STEM_RE)
    df["stem"] = df["basename"].str.replace(".bmp", "", regex=False)
    df["burst"] = parsed["burst"].astype("Int64")
    df["date"] = pd.to_datetime(df["filename_date"], format="%Y%m%d", errors="coerce")
    df["hour"] = df["filename_time"].str[:2].astype(int)
    return df


def scan_images(paths: list[Path]) -> pd.DataFrame:
    """이미지 헤더만 읽어 해상도·모드·파일크기·깨짐 여부를 반환한다.

    `Image.open` 후 `.size`/`.mode`만 읽고, 별도로 `.verify()`만 호출한다(`load()` 없음).
    """
    rows = []
    for p in paths:
        row = {"path": str(p), "name": p.name, "ext": p.suffix.lower(), "filesize": p.stat().st_size}
        try:
            with Image.open(p) as im:
                row["width"], row["height"] = im.size
                row["mode"] = im.mode
        except Exception as e:  # noqa: BLE001
            row["width"] = row["height"] = row["mode"] = None
            row["header_error"] = str(e)
        try:
            with Image.open(p) as im2:
                im2.verify()
            row["corrupted"] = False
        except Exception as e:  # noqa: BLE001
            row["corrupted"] = True
            row["verify_error"] = str(e)
        rows.append(row)
    return pd.DataFrame(rows)


def intensity_stats(paths: list[Path], marker_min_index: int = MARKER_MIN_INDEX) -> pd.DataFrame:
    """장당 픽셀 강도 통계.

    BMP/JPG 모두 8bit 팔레트(mode 'P')이며 팔레트 인덱스 0~`marker_min_index`-1은
    항등 그레이스케일, 그 이상은 오버레이 마커용 예약 컬러라 실제 밝기가 아니다.
    통계는 마커 인덱스를 제외한 픽셀에서 계산하고, 마커 픽셀 비율은 별도 컬럼으로 낸다.
    """
    rows = []
    for p in paths:
        with Image.open(p) as im:
            a = np.asarray(im)
        gray_mask = a < marker_min_index
        g = a[gray_mask].astype(np.float64)
        n = a.size
        row = {
            "path": str(p), "name": p.name,
            "mean": g.mean() if g.size else np.nan,
            "std": g.std() if g.size else np.nan,
            "min": int(g.min()) if g.size else np.nan,
            "max": int(g.max()) if g.size else np.nan,
            "p01": np.percentile(g, 1) if g.size else np.nan,
            "p99": np.percentile(g, 99) if g.size else np.nan,
            "frac_sat_low": float((g == 0).sum()) / n,
            "frac_sat_high": float((g == marker_min_index - 1).sum()) / n,
            "marker_frac": float((~gray_mask).sum()) / n,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _find_raw_bmp(root: Path, stem: str) -> Path | None:
    for m in MACHINES:
        p = raw_bmp_dir(root) / m / f"{stem}.bmp"
        if p.exists():
            return p
    return None


def bmp_jpg_psnr(root: Path) -> pd.DataFrame:
    """라벨 세트 JPG와 동일 이름 원본 BMP를 비교(해상도 일치·PSNR)."""
    from skimage.metrics import peak_signal_noise_ratio as psnr

    rows = []
    for jp in labeled_image_paths(root):
        stem = jp.stem
        bp = _find_raw_bmp(root, stem)
        row = {"stem": stem, "has_bmp": bp is not None}
        if bp is not None:
            with Image.open(bp) as bi:
                b = np.asarray(bi)
            with Image.open(jp) as ji:
                j = np.asarray(ji)
            row["bmp_size"] = bi.size
            row["jpg_size"] = ji.size
            row["size_match"] = bi.size == ji.size
            if row["size_match"]:
                row["psnr"] = psnr(b.astype(np.float64), j.astype(np.float64), data_range=255)
            else:
                row["psnr"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_yolo_labels(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """YOLO txt 라벨 파싱.

    반환: (파일별 요약 DataFrame, 박스별 레코드 DataFrame).
    형식 오류: 열 개수 != 5, class != 0, 좌표가 [0,1] 밖, w 또는 h <= 0.
    """
    files = sorted((labeled_dir(root) / "labels").glob("*.txt"))
    summary_rows, box_rows = [], []
    for p in files:
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_bad = 0
        for li, line in enumerate(lines):
            parts = line.split()
            bad = len(parts) != 5
            if not bad:
                try:
                    cls, cx, cy, w, h = (float(x) for x in parts)
                except ValueError:
                    bad = True
                else:
                    bad = (
                        cls != 0
                        or not (0.0 <= cx <= 1.0)
                        or not (0.0 <= cy <= 1.0)
                        or w <= 0 or h <= 0
                        or cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6
                        or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6
                    )
                    box_rows.append({
                        "stem": p.stem, "box_idx": li, "cls": cls,
                        "cx": cx, "cy": cy, "w": w, "h": h, "valid": not bad,
                    })
            if bad:
                n_bad += 1
        summary_rows.append({"stem": p.stem, "n_boxes": len(lines), "n_format_errors": n_bad})
    return pd.DataFrame(summary_rows), pd.DataFrame(box_rows)


def parse_voc_xml(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PASCAL VOC xml 파싱(참고용). 반환: (파일별 요약, 박스별 레코드 — cx/cy/w/h는 정규화값)."""
    files = sorted((labeled_dir(root) / "voc_xml").glob("*.xml"))
    summary_rows, box_rows = [], []
    for p in files:
        try:
            r = ET.parse(p).getroot()
            width = int(r.find("size/width").text)
            height = int(r.find("size/height").text)
            objs = r.findall("object")
            for oi, obj in enumerate(objs):
                b = obj.find("bndbox")
                xmin, ymin, xmax, ymax = (int(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
                box_rows.append({
                    "stem": p.stem, "box_idx": oi, "name": obj.findtext("name"),
                    "width": width, "height": height,
                    "cx": (xmin + xmax) / 2 / width, "cy": (ymin + ymax) / 2 / height,
                    "w": (xmax - xmin) / width, "h": (ymax - ymin) / height,
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                })
            summary_rows.append({"stem": p.stem, "n_boxes": len(objs), "width": width, "height": height})
        except Exception as e:  # noqa: BLE001
            summary_rows.append({"stem": p.stem, "n_boxes": None, "width": None, "height": None, "error": str(e)})
    return pd.DataFrame(summary_rows), pd.DataFrame(box_rows)


def bbox_geometry(box_df: pd.DataFrame, size_lookup: dict[str, tuple[int, int]], eps: float = 0.02) -> dict:
    """bbox 중심 클러스터링(DBSCAN)·픽셀 크기·완전일치 탐지.

    box_df: stem, box_idx, cx, cy, w, h 컬럼 필요(parse_yolo_labels의 두 번째 반환값).
    size_lookup: stem -> (width, height) px, 정규화 bbox를 픽셀 크기로 환산하는 데 쓴다.
    """
    from sklearn.cluster import DBSCAN

    df = box_df.copy()
    wh = df["stem"].map(size_lookup)
    df["img_w"] = wh.map(lambda t: t[0] if t else np.nan)
    df["img_h"] = wh.map(lambda t: t[1] if t else np.nan)
    df["px_w"] = df["w"] * df["img_w"]
    df["px_h"] = df["h"] * df["img_h"]

    coords = df[["cx", "cy"]].to_numpy()
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(coords)
    df["cluster"] = labels
    vc = df.loc[df["cluster"] >= 0, "cluster"].value_counts()
    n_clusters = int((vc.index >= 0).sum())
    top5_share = float(vc.head(5).sum() / len(df)) if len(df) else np.nan
    noise_frac = float((labels == -1).sum() / len(df)) if len(df) else np.nan

    # 좌표 완전 일치(소수 6자리) 쌍
    key = df[["cx", "cy", "w", "h"]].round(6).astype(str).agg("|".join, axis=1)
    dup_box_pairs = int(key.duplicated().sum())

    # bbox 세트 통째로 동일한 이미지 쌍(같은 stem 내 정렬된 box 좌표 집합 비교)
    set_key = (
        df.sort_values(["stem", "cx", "cy"])
        .groupby("stem")
        .apply(lambda g: "|".join(g[["cx", "cy", "w", "h"]].round(6).astype(str).agg(",".join, axis=1)))
    )
    dup_image_sets = int(set_key.duplicated().sum())

    return {
        "box_df": df,
        "n_clusters": n_clusters,
        "cluster_sizes": vc.to_dict(),
        "top5_share": top5_share,
        "noise_frac": noise_frac,
        "dup_box_pairs": dup_box_pairs,
        "dup_image_sets": dup_image_sets,
        "n_boxes": len(df),
        "n_images": df["stem"].nunique(),
        "median_px_w": float(df["px_w"].median()),
        "median_px_h": float(df["px_h"].median()),
    }


def label_coverage(index_df: pd.DataFrame, label_stems: set[str]) -> dict:
    """라벨 500장이 호기·일자별로 전체(2,532) 대비 편중됐는지 카이제곱 검정."""
    from scipy.stats import chi2_contingency

    df = index_df.copy()
    df["labeled"] = df["stem"].isin(label_stems)

    machine_tab = pd.crosstab(df["machine"], df["labeled"])
    chi2_m, p_m, dof_m, _ = chi2_contingency(machine_tab)

    df["date_bin"] = df["date"].dt.to_period("M").astype(str)
    date_tab = pd.crosstab(df["date_bin"], df["labeled"])
    chi2_d, p_d, dof_d, _ = chi2_contingency(date_tab)

    return {
        "machine_table": machine_tab,
        "machine_chi2": chi2_m, "machine_p": p_m, "machine_dof": dof_m,
        "date_table": date_tab,
        "date_chi2": chi2_d, "date_p": p_d, "date_dof": dof_d,
        "n_labeled": int(df["labeled"].sum()), "n_total": len(df),
    }


def overlay_boxes(root: Path, index_df: pd.DataFrame, dilate: int = 2, min_size: int = 8,
                   max_aspect: float = 5.0,
                   label_boxes: pd.DataFrame | None = None,
                   size_lookup: dict[str, tuple[int, int]] | None = None,
                   marker_min_index: int = MARKER_MIN_INDEX) -> tuple[pd.DataFrame, pd.DataFrame]:
    """검사기가 픽셀에 새겨 넣은 오버레이 박스(팔레트 인덱스 244~255) 검출.

    팔레트 인덱스 `marker_min_index` 이상인 픽셀을 마스크로 잡아 `dilate`회 팽창한 뒤
    연결요소로 묶고, 각 연결요소의 바운딩박스를 검사기의 탐지 박스 후보로 본다.
    `min_size`보다 폭·높이가 모두 작은 연결요소(팽창 후 남는 잡음)는 버린다.
    또한 (긴 변/짧은 변) 비율이 `max_aspect`를 넘는 연결요소는 버린다 — 실측 결과 진짜
    탐지 박스는 이 비율이 최대 2.18인 반면, 이미지 왼쪽 테두리에 붙은 6x332px 세로줄
    (105건, 전부 machine_1, 비율 55.3)처럼 탐지 박스가 아닌 테두리/눈금 아티팩트가
    따로 존재해 간격 없이 분리된다.

    반환: (박스별 DataFrame — stem/machine/xmin·ymin·xmax·ymax/width/height/indices,
    이미지별 요약 DataFrame — stem/machine/n_overlay_boxes/indices_used).
    `label_boxes`(parse_yolo_labels의 두 번째 반환값)와 `size_lookup`(stem -> (width,height)
    px, bbox_geometry와 동일한 형식)을 함께 주면, YOLO bbox 중심이 오버레이 박스 내부에
    들어가는지(`n_labels`, `n_inside`)도 이미지별 요약에 추가한다.
    """
    from scipy import ndimage

    stem_machine = index_df.drop_duplicates(subset="stem", keep="first").set_index("stem")["machine"]

    box_rows, summary_rows = [], []
    for stem, machine in stem_machine.items():
        p = raw_bmp_dir(root) / machine / f"{stem}.bmp"
        if not p.exists():
            continue
        with Image.open(p) as im:
            arr = np.asarray(im)
        mask = arr >= marker_min_index
        dilated = ndimage.binary_dilation(mask, iterations=dilate) if mask.any() else mask
        labeled_arr, n_labels = ndimage.label(dilated)
        boxes = []
        if n_labels:
            for lbl, sl in enumerate(ndimage.find_objects(labeled_arr), start=1):
                if sl is None:
                    continue
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop
                w, h = x1 - x0, y1 - y0
                if w < min_size and h < min_size:
                    continue
                if max(w, h) / max(1, min(w, h)) > max_aspect:
                    continue
                comp_mask = (labeled_arr == lbl) & mask
                idxs = tuple(sorted(int(v) for v in np.unique(arr[comp_mask])))
                boxes.append({
                    "stem": stem, "machine": machine, "box_idx": len(boxes),
                    "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
                    "width": w, "height": h, "indices": idxs,
                    "n_marker_px": int(comp_mask.sum()),
                })
        box_rows.extend(boxes)
        pooled_indices = tuple(sorted({v for b in boxes for v in b["indices"]}))
        summary_rows.append({
            "stem": stem, "machine": machine, "n_overlay_boxes": len(boxes),
            "indices_used": pooled_indices,
        })

    box_df = pd.DataFrame(box_rows)
    summary_df = pd.DataFrame(summary_rows)

    if label_boxes is not None and size_lookup is not None and len(box_df):
        boxes_by_stem = {s: g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
                         for s, g in box_df.groupby("stem")}
        n_labels_map, n_inside_map = {}, {}
        for stem, g in label_boxes.groupby("stem"):
            obxs = boxes_by_stem.get(stem)
            n_labels_map[stem] = len(g)
            wh = size_lookup.get(stem)
            if obxs is None or not len(obxs) or wh is None:
                n_inside_map[stem] = 0
                continue
            img_w, img_h = wh
            px = g["cx"].to_numpy() * img_w
            py = g["cy"].to_numpy() * img_h
            inside = np.zeros(len(px), dtype=bool)
            for xmin, ymin, xmax, ymax in obxs:
                inside |= (px >= xmin) & (px < xmax) & (py >= ymin) & (py < ymax)
            n_inside_map[stem] = int(inside.sum())
        summary_df["n_labels"] = summary_df["stem"].map(n_labels_map).fillna(0).astype(int)
        summary_df["n_inside"] = summary_df["stem"].map(n_inside_map).fillna(0).astype(int)

    return box_df, summary_df


def near_duplicates(root: Path, index_df: pd.DataFrame, sample_per_machine: int = 300,
                     seed: int = 0) -> pd.DataFrame:
    """시간상 인접한 촬영 프레임 간 SSIM.

    최초 가정(파일명 `(k)`가 같은 초 안의 연사 프레임 번호)은 데이터로 반박됐다: 같은
    호기+날짜+HHMMSS로 묶이는 그룹은 2,530개 중 2개뿐이고, 같은 호기·날짜 안에서 정렬한
    인접 촬영 간 시간 간격은 중앙값 4초(균일한 검사 주기로 추정)다. `(k)`는 0~9가 거의
    균등분포라 초당 연사가 아니라 롤링 프레임 카운터로 보인다(추정). 대신 같은 호기+날짜
    안에서 촬영 시각(시간+`(k)`) 순으로 정렬한 **인접 프레임 쌍**의 SSIM을 잰다 — 이것이
    랜덤 split 시 실제로 train/test에 갈라질 수 있는 "거의 같은 영상" 인접쌍이다.
    전수 계산이 느리면 호기별 `sample_per_machine`쌍으로 샘플링한다(그 경우 결과는
    "(추정)"으로 표시할 것).
    """
    from skimage.metrics import structural_similarity as ssim

    df = index_df.sort_values(["machine", "filename_date", "filename_time", "burst"]).copy()
    rows = []
    rng = np.random.default_rng(seed)
    for machine, g in df.groupby("machine"):
        pairs = []
        for _, gg in g.groupby("filename_date"):
            stems = gg["stem"].tolist()
            for a, b in zip(stems[:-1], stems[1:]):
                pairs.append((a, b))
        sampled = pairs
        estimated = False
        if len(pairs) > sample_per_machine:
            idx = rng.choice(len(pairs), size=sample_per_machine, replace=False)
            sampled = [pairs[i] for i in idx]
            estimated = True
        for a, b in sampled:
            pa, pb = _find_raw_bmp(root, a), _find_raw_bmp(root, b)
            if pa is None or pb is None:
                continue
            with Image.open(pa) as ia, Image.open(pb) as ib:
                arra, arrb = np.asarray(ia), np.asarray(ib)
            if arra.shape != arrb.shape:
                continue
            s = ssim(arra, arrb, data_range=255)
            rows.append({"machine": machine, "a": a, "b": b, "ssim": s, "estimated": estimated})
    return pd.DataFrame(rows)
