# KAMP — 제6회 K-인공지능 제조데이터 분석 경진대회

- 과제 기간: 2026-09-21 ~ **2026-10-08(목) 23:59**
- 주제: **③ 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석** (선정 근거: [docs/topic_selection.md](docs/topic_selection.md))
- 현재 단계: 데이터 품질 진단 완료 → 모델링 착수

## 폴더 구조

```
KAMP/
├─ README.md
├─ CLAUDE.md               # 팀 공유 규칙 (커밋·브랜치·PR 규칙, 심사 기준, 작업 도메인)
├─ requirements.txt        # Python 3.12 기준 고정 버전
├─ .github/                # PR 템플릿
├─ data/                   # 원본 zip + raw/·processed/ (전부 git 제외) — data/README.md
├─ notebooks/              # 실행·결과 기록 (git 포함, 실행 결과 포함)
│  ├─ _template.ipynb      #   새 노트북은 이걸 복사해서 시작
│  └─ scratch/             #   개인 실험용 (git 제외)
├─ src/                    # 재사용 코드 (.py) — 노트북이 import
├─ figures/<노트북명>/     # 노트북이 저장한 그림 (git 포함)
├─ results/<노트북명>/     # 지표·오류 사례 CSV (git 포함) — results/README.md
├─ reports/                # 노트북별 md 요약 (git 포함)
├─ models/                 # 학습된 모델 가중치 (git 제외)
└─ docs/                   # 제출물 초안(보고서·발표자료), 회의 메모, 주제 선정 기록
```

## 시작하기

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
# data/ 에 "3. 소성가공 예지보전 AI 데이터셋.zip"을 넣은 뒤 (data/README.md 참고)
python src/extract.py                               # data/raw/ 로 해제 (노트북에서도 자동 수행)
cd notebooks
PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 01_data_quality.ipynb
PYTHONUTF8=1 jupyter nbconvert --to notebook --execute --inplace 02_diagnosis_deep.ipynb
```

- 노트북은 `notebooks/` 안에서 실행한다는 전제로 경로를 잡는다 (`ROOT = cwd.parent`, `sys.path`에 `src/` 추가).
- Windows에서 노트북 실행 시 한글 출력 깨짐 방지: `PYTHONUTF8=1` 환경변수 설정.

## 폴더별 작업 방법

한 가지 작업 = 노트북 1개 + (필요 시) `src/` 함수 + 리포트 md 1개. 흐름은 **`src/`에 함수 작성 → 노트북에서 호출·실행 → 결과를 `figures/`·`results/`에 저장 → `reports/`에 요약**.

| 폴더 | 무엇을 두나 | 규칙 |
|---|---|---|
| `notebooks/` | 실행 순서와 결과 | 파일명 `<번호>_<내용>_<이니셜>.ipynb`. 번호 대역: **0x 진단 · 1x 전처리·피처 · 2x 모델 · 3x 분석(오류·영향요인·현장 활용)**. 예: `11_window_features_CHS.ipynb`, `21_model_iforest_CHS.ipynb`. `_template.ipynb`를 복사해 시작하고 셀 안의 `NB` 값을 파일명과 같게 바꾼다. 긴 로직은 노트북에 쓰지 말고 `src/`로 뺀다. (`01`·`02`는 규칙 도입 전 산출물이라 이름 유지) |
| `notebooks/scratch/` | 개인 탐색·실험 | git 제외. 자유롭게 쓰고, 공유할 결과가 나오면 정식 번호 노트북으로 옮긴다. |
| `src/` | 재사용 함수 (.py) | 역할별 파일로 나눈다: `preprocess.py`(DC 제거·형식 통일) / `split.py`(세그먼트 GroupKFold·시간 블록) / `features.py` / `models.py` / `evaluate.py`(지표). 경로는 `paths.py`(`ROOT`, `DATA_RAW`, `DATA_PROCESSED`, `MODELS` …)만 쓴다. 공통 파일(`preprocess`·`split`·`paths`)은 변경 시 작은 PR로 따로 올린다. |
| `figures/<노트북명>/` | 그림 png | `FIG, RES = paths.nb_dirs(NB)` 후 `paths.save_fig(FIG, "이름.png")`로만 저장 → 노트북마다 폴더가 나뉘어 파일명이 겹치지 않는다. |
| `results/<노트북명>/` | 지표·오류 사례 CSV | 모델 노트북은 `RES / "metrics.csv"`에 공통 컬럼으로 저장 (`results/README.md`). 비교표 `results/model_comparison.csv`는 3x 노트북이 모아서 생성한다. |
| `data/processed/` | 단계 간 주고받는 가공 데이터 | git 제외. 파일명 앞에 만든 노트북 번호(`11_…parquet`). 새 파일은 `data/README.md` 표에 등록해 입력 계약으로 삼는다. |
| `reports/` | 노트북별 요약 md | 파일명은 노트북과 같게(`21_model_iforest_CHS.md`). 발견 → 시사점 → 활용 아이디어 순. 그림은 `../figures/<노트북명>/…` 경로로 링크. |
| `models/` | 모델 가중치 | git 제외. `models/<노트북명>/`에 저장하고, 재학습 코드로 재현 가능하게 한다. |

## 동시 작업 규칙 (3인)

- **노트북 1개 = 담당자 1명.** 남의 노트북은 직접 고치지 말고 PR 리뷰 코멘트로 요청한다(노트북 JSON은 병합이 안 된다).
- 노트북·그림·결과는 파일명과 폴더에 번호와 이니셜이 들어가므로 서로 겹치지 않는다. 겹칠 수 있는 곳은 `src/` 공통 파일과 `data/README.md` 표뿐 → 작업 전 `main`을 pull 하고, 공통 파일 변경은 작은 PR로 먼저 병합한다.
- 브랜치 하나에는 노트북 1~2개 단위로 작업하고 자주 병합한다. 오래 열어 둔 브랜치는 `main`을 merge해 최신화한다.

## 작업 규칙

- 브랜치: `main`(제출 기준) / 작업은 `<type>/<이니셜 3글자>-<작업내용>` 브랜치 → PR. type은 `feat/ fix/ docs/ test/ chore/ refactor/`. (예: `feat/CHS-model-baseline`, `chore/INFRA-root-restructure`) PR 제목은 `[Data]`/`[Model]`/`[Analysis]`/`[Docs]`/`[Chore]` 구분 후 명사형 설명, merge commit으로 병합. 상세는 `CLAUDE.md` 참고.
- 노트북은 **실행 결과를 포함해서** 커밋한다 (팀원이 안 돌려도 결과를 볼 수 있게). 단, 출력이 5MB를 넘으면 figures로 빼고 출력은 지운다.
- 원본 데이터(`data/`의 zip, `data/raw/`)는 절대 수정하지 않는다. 가공 결과는 `data/processed/`에 별도 파일로.
- 제출물에는 소속·로고 등 식별 정보를 넣지 않는다 (블라인드 평가).

## 심사 기준 (100점)

| 문항 | 배점 |
|---|---|
| 1. 데이터 이해 및 진단 | 15 |
| 2. AI 예측모델 개발 (베이스라인 포함 2개 이상 비교) | 40 |
| 3. 영향요인 및 오류분석 (FN/FP 집중 조건) | 15 |
| 4. 현장 활용방안 | 10 |
| 5. 창의성·차별성 | 10 |
| 6. 코드 및 재현성 | 10 |

## 진단 결과

- [01 데이터 품질 진단](reports/01_data_quality.md) — 파일 구성, burst 세그먼트(정상 599 / 이상 21), 전류 에일리어싱, 라벨 노이즈 의심 구간
- [02 심층 진단](reports/02_diagnosis_deep.md) — 파일 형식 누수(|DC| AUC 0.9443), 채널별 세그먼트 등급, 운전 상태, 윈도우 민감도, 탐지 지연
