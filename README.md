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

# 모델 훈련 코드 실행

딥러닝 모델을 훈련하고자 할 경우 (코드를 수정하여 임의의 전처리 시퀀스, 모델 사용 가능)

```
python run_fit_dl.py
```

# ML 모델 검증 코드 실행

Features를 선택하여 ML 이상탐지 모델을 검증할 경우 (코드를 수정하여 임의의 모델, 특성추출, ML 모델 사용 가능)

```
python run_fit_ml.py
```

# 사전훈련된 모델

사전훈련된 모델의 가중치는 다음 디렉토리에 저장

```
.result/
    ㄴ 1dcae --> 직교좌표 1D CAE
    ㄴ 1dcae_polar --> 직교+극좌표 1D CAE
    ㄴ lstm1dcae --> 직교좌표 1D LSTM CAE
    ㄴ lstm1dcae_polar --> 직교+극좌표 1D LSTM CAE
    ㄴ stft2dcae --> STFT 2D CAE
```