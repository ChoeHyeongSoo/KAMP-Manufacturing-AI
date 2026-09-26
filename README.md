# KAMP — 제6회 K-인공지능 제조데이터 분석 경진대회

- 과제 기간: 2026-09-21 ~ **2026-10-08(목) 23:59**
- 주제: **③ 진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 오경보 분석** (선정 근거: [docs/topic_selection.md](docs/topic_selection.md))
- 현재 단계: 데이터 품질 진단 완료 → 모델링 착수

## 폴더 구조

```
KAMP/
├─ README.md
├─ CLAUDE.md               # 팀 공유 규칙 (커밋·브랜치·PR 규칙, 심사 기준, 작업 도메인) — 항상 최신으로 유지
├─ requirements.txt        # Python 3.12 기준 고정 버전
├─ .github/                # PR 템플릿
├─ data/                   # 원본 zip + raw/·processed/ (전부 git 제외) — data/README.md 참고
├─ notebooks/              # 번호 순서: 01_data_quality → 02_diagnosis_deep → 03_features → 04_model_baseline → 05_model_<이름> → 06_error_analysis
├─ src/                    # 노트북에서 import하는 재사용 코드 (extract, data_quality, segments, signal_checks …)
├─ figures/                # 노트북이 저장한 png (git 포함)
├─ reports/                # 노트북별 md 요약 — 발견 → 시사점 → 활용 아이디어 (git 포함)
├─ results/                # 모델 비교표·fold별 점수·오류 사례 CSV (git 포함)
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
