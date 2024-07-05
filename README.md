# KSNVE-AI-Challenge
제 1회 한국소음진동공학회 AI 챌린지 (2024) - RTES 연구실 참여 <br>
**공식 홈페이지** :📒 [한국소음진동공학회 AI 챌린지](https://ksnve.notion.site/1-AI-2024-1101af726e694d60867d32fa0e6ab53e)<br>
**Project Workspace** :📒 [Team Project Notion](https://www.notion.so/skipper0527/AI-4bd41e7a934b4329960bb453665150ec?pvs=4)<br>

# 사전 준비사항

## 데이터셋 다운로드

데이터의 용량이 높으므로 git 저장소는 원본 데이터를 포함하지 않는다. `dataset` 폴더를 만든 후 그 안에 AI 챌린지 홈페이지에서 다운받은 데이터의 `train`, `eval` 폴더를 위치시킨다.

```
.dataset/
    ㄴ train/
        ㄴ ...
    ㄴ eval/
        ㄴ ...
```

## 의존성 라이브러리 설치

```
pip install -r requirements.txt
```

# 테스트용 코드 실행

테스트용 코드 및 모델 코드를 변형시켜가며 테스트 수행 요망

```
python run.py
```