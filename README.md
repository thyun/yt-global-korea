# 외국인의 한국 생활 — YouTube 동영상 제작 에이전트

6개의 AI 에이전트가 순차적으로 협업하여 8분 분량의 YouTube 동영상 제작 자료를 자동 생성합니다.

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. API 키 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 입력

# 3. 새 에피소드 생성 (주제 자동 선정 + 전 단계 실행)
python run.py
```

## 에이전트 파이프라인

```
1. Research Agent   → YouTube 인기 동영상 검색 & 랭킹 (소재 발굴)
2. Topic Agent      → 랭킹 결과 분석 후 에피소드 주제 선정
3. Script Agent     → 8분 분량 한국어 스크립트 작성
4. Review Agent     → 스크립트 품질 검토 및 개선
5. Asset Agent      → 씬별 영상/이미지 프롬프트 생성
6. Production Agent → 최종 편집 및 제작 가이드
```

## 산출물 구조

```
episodes/
└── ep001_제목슬러그/
    ├── 01_research/videos.json       ← YouTube 랭킹 결과
    ├── 02_topic/topic.json           ← 선정된 주제
    ├── 03_script/script.md
    ├── 04_review/script_revised.md
    ├── 05_assets/assets.json
    └── 06_production/production_guide.md
```

## 주요 옵션

```bash
# 특정 단계만 실행
python run.py --step script --episode ep001

# 특정 단계부터 이어서 실행
python run.py --from-step review --episode ep001

# 설정 파일 지정
python run.py --config config/settings.yaml
```

## 설정 변경

`config/settings.yaml`에서 LLM 모델, 목표 단어 수 등을 조정할 수 있습니다.
