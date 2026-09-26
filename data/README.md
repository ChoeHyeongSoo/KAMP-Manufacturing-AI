# data/ — 원본 데이터셋 (git 제외)

KAMP 포털 공지사항(https://www.kamp-ai.kr/noticeList)에서 대회용 제조AI데이터셋을 받아 이 폴더에 **파일명 그대로** 둔다.

```
data/
├─ 1. 사출성형기 AI 데이터셋.zip          (약 35 MB, CSV 4개)
├─ 2. 용접기 AI 데이터셋.zip              (후보 아님)
├─ 3. 소성가공 예지보전 AI 데이터셋.zip   (약 1.3 MB, CSV 2개)
├─ 4. X-ray 검사장비 AI 데이터셋.zip
└─ 5. 자원 최적화 AI 데이터셋.zip         (후보 아님)
```

- zip 안의 폴더명이 한글이라 `zipfile`로 열 때 파일명이 cp437→cp949로 깨진다. 각 주제의 진단 노트북은 `os.path.basename`으로 파일만 꺼내 `topics/<주제>/data_raw/`에 푼다.
- 원본은 수정하지 않는다.
