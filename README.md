# KAMP — 제6회 K-인공지능 제조데이터 분석 경진대회

- 과제 기간: 2026-09-21 ~ **2026-10-08(목) 23:59**
- 후보 주제: ① 사출성형 품질불량 예측 / ③ 프레스 유압펌프 이상 조기탐지 / ④ X-ray 이물질 탐지
- 현재 단계: 주제 선정을 위한 데이터 품질 진단 (①, ③ 완료, ④ 미착수)

## 폴더 구조

```
KAMP/
├─ README.md
├─ CLAUDE.md                 # 팀 공유 규칙 (커밋 규칙·심사 기준·작업 도메인) — 항상 최신으로 유지
├─ requirements.txt          # Python 3.12 기준 고정 버전
├─ .gitignore
├─ .claude/                  # 개인 작업일지·로컬 설정 (git 제외, 팀 공유 안 됨)
├─ data/                     # 원본 zip (git 제외) — data/README.md의 다운로드 안내 참고
├─ common/                   # 주제 공통 유틸 (평가지표, 분할, 플롯 스타일) — 주제 확정 후 채움
├─ docs/                     # 제출물 초안(보고서·발표자료), 회의 메모
└─ topics/
   ├─ 01_injection/          # 주제 ① 사출성형
   ├─ 03_press_pump/         # 주제 ③ 유압펌프 시계열
   └─ 04_xray/               # 주제 ④ X-ray 영상
      ├─ README.md           # 주제 요약 + 산출물 목록
      ├─ data_raw/           # zip 해제본 (git 제외)
      ├─ notebooks/          # 번호 순서: 01_data_quality → 02_eda → 03_features → 04_model → 05_error_analysis
      ├─ src/                # 노트북에서 import하는 재사용 코드
      ├─ figures/            # 노트북이 저장한 png (git 포함)
      └─ reports/            # 노트북 없이 읽는 md 요약 (git 포함)
```

## 시작하기

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
# data/ 에 zip 5개를 넣은 뒤 (data/README.md 참고)
cd topics/01_injection/notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_quality.ipynb
```

- 노트북은 `notebooks/` 안에서 실행한다는 전제로 경로를 잡는다 (`ROOT = cwd.parent`).
- zip 해제는 각 주제 노트북/스크립트가 `data/*.zip → topics/<주제>/data_raw/`로 수행한다. `data_raw/`는 커밋하지 않는다.
- Windows에서 노트북 실행 시 한글 출력 깨짐 방지: `PYTHONUTF8=1` 환경변수 설정.

## 작업 규칙

- 브랜치: `main`(제출 기준) / 작업은 `<type>/<이니셜 3글자>-<작업내용>` 브랜치 → PR. type은 `feat/ fix/ docs/ test/ chore/ refactor/`. (예: `feat/CHS-01-injection-data-quality`, `docs/CHS-topic-selection-meeting`) 상세는 `CLAUDE.md` 참고.
- 노트북은 **실행 결과를 포함해서** 커밋한다 (팀원이 안 돌려도 결과를 볼 수 있게). 단, 출력이 5MB를 넘으면 figures로 빼고 출력은 지운다.
- 원본 데이터(`data/`, `data_raw/`)는 절대 수정하지 않는다. 가공 결과는 별도 파일로.
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

## 진단 결과 바로가기

- [주제 ① 사출성형 품질 진단](topics/01_injection/reports/01_data_quality.md)
- [주제 ③ 유압펌프 품질 진단](topics/03_press_pump/reports/01_data_quality.md)
