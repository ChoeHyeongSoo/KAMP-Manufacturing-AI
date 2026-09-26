# data/ — 원본 데이터셋 (git 제외)

KAMP 포털 공지사항(https://www.kamp-ai.kr/noticeList)에서 대회용 제조AI데이터셋을 받아 이 폴더에 **파일명 그대로** 둔다.
이 저장소는 주제 ③ 데이터만 사용한다.

```
data/
├─ 3. 소성가공 예지보전 AI 데이터셋.zip   (약 0.4 MB, CSV 2개) — 원본, 수정 금지
├─ raw/          # zip 해제본 (press_data_normal.csv, outlier_data.csv)
└─ processed/    # 가공 결과 (예: segments.csv) — 코드로 재생성
```

- `raw/`는 `python src/extract.py`로 만든다. 노트북에서 `data_quality.load()`를 호출할 때 파일이 없으면 자동으로 해제한다.
- `raw/`, `processed/`, zip은 모두 git 제외. 가공 결과는 원본을 덮어쓰지 않고 `processed/`에 별도 파일로 남긴다.
