# 주제 ④ X-ray 영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석

데이터: `data/4. X-ray 검사장비 AI 데이터셋.zip` (약 2.4 GB 해제 기준) — **1차 데이터 품질 진단 완료**(`reports/01_data_quality.md`). 모델링은 아직 착수 전.

## 데이터 준비

```bash
python topics/04_xray/src/extract.py      # 이미 풀린 파일은 건너뜀, --force 로 재해제
```

zip 내부는 실습용 구성(참고 코드·가중치·중복 이미지 세트 혼재)이라 용도별로 나눠 푼다. `data_raw/`는 git 제외.

```
data_raw/
├─ raw_bmp/machine_{1,2,3}/   원본 BMP 2,532장 (1호기 943 / 2호기 805 / 3호기 784), 2020-06-22 ~ 09-22
├─ raw_bmp/index.csv          파일별 호기·시리얼·폴더 날짜·촬영 시각·zip 내 중복 수·CRC
├─ labeled/images/            라벨 대응 JPG 400장
├─ labeled/labels/            YOLO txt 500개 (class 0 = defect, 정규화 cx cy w h)
├─ labeled/voc_xml/           라벨링 도구(OpenLabeling) 출력 PASCAL VOC xml 833개 (참고)
└─ reference/                 참고 yolov3 코드·train/test 목록(15장)·학습 로그, 가중치 *.pt 7개(1.7 GB), 라벨링 도구 소스
```

## 구성 파악 결과 (2026-09-26, 목록·파일명 기준)

- 원본 폴더명이 `<호기>/<시리얼>_<날짜>_NgImage/` → **2,532장 전부 장비가 NG로 판정한 영상** (추정). OK 영상은 없다.
- zip 안에 같은 파일이 인접 날짜 폴더에 두 번 들어간 경우 277건, 전부 바이트 동일. 평탄화하며 하나로 합쳤고 `index.csv`의 `n_copies`로 확인 가능. 파일명 스템이 호기 간에 겹치는 경우 3건.
- 파일명 `NNN_YYYYMMDD_HHMMSS(k).bmp`. `(k)`는 같은 초 안의 연사 번호가 아니라(같은 초 그룹은 2,530개 중 2개뿐) 롤링 프레임 카운터로 보인다(추정). 실제 인접 촬영 간격은 중앙값 4초.
- 라벨 500개는 모두 원본 BMP와 대응. JPG는 그중 400장만 제공(`images 15 ⊂ 50 ⊂ … ⊂ 400` 실습 세트의 합집합) — 단 이 `.jpg`는 매직바이트가 `BM`인 BMP 원본과 완전 동일 파일(PSNR 무한대 400/400).
- 라벨은 단일 클래스(`defect`), bbox 픽셀 크기 중앙값 10x10px로 **매우 작은 객체**. bbox 중심의 DBSCAN 클러스터는 4개뿐이고 상위 4개(=상위5) 점유율 **97.99%**.
- **확정된 최대 리스크(라벨 누수)**: 원본 BMP의 팔레트 인덱스 244~255는 잡음이 아니라 **검사기 소프트웨어가 그려 넣은 탐지 박스**다(호기 3대 전체에서 검출, 이미지당 0~3개, 픽셀 크기 중앙값 22x22px). YOLO 라벨 bbox 중심 1,147개 중 **1,145개(99.83%)가 이 오버레이 박스 내부**에 있다 — 즉 라벨은 사람이 새로 찾은 이물이 아니라 **검사기 판정을 그대로 옮겨 적은 것**이다. 그대로 학습하면 모델은 이물이 아니라 오버레이 박스(22px 정사각 테두리)를 찾는 법을 배운다. 인페인팅으로 오버레이를 지워도 "검사기가 놓친 이물"은 이 데이터에 없어, ④로 가능한 문제는 "이물 탐지"가 아니라 "검사기 판정 재현"뿐이다. **④를 주제로 채택할지 재논의가 필요한 최대 리스크.**
- 가이드북·변수 설명 없음 (추정). 참고 노트북 `reference/yolov3_20201200.ipynb`.

## 산출물

- `src/data_quality.py` — `scan_images`, `intensity_stats`, `bmp_jpg_psnr`, `parse_yolo_labels`, `parse_voc_xml`, `bbox_geometry`, `overlay_boxes`, `label_coverage`, `near_duplicates`
- `notebooks/01_data_quality.ipynb` — 전체 실행 결과 포함(재실행: `jupyter nbconvert --to notebook --execute --inplace`)
- `figures/*.png` — 해상도·촬영시계열·강도·PSNR·bbox 히트맵/크기/오버레이·검사기 오버레이 박스 대조·SSIM 등 11장
- `reports/01_data_quality.md` — 진단 결과·수치·필요 처리 표·주제 채택 총평

## 다음 단계

`reports/01_data_quality.md`의 "라벨 신뢰도" 리스크를 팀 회의에서 공유하고 ①·③ 대비 주제 우선순위를 재논의. 채택 시 §6(다음 단계 인사이트)의 전처리·검증 전략부터 착수.
