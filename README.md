# KSNVE-AI-Challenge
제 1회 한국소음진동공학회 AI 챌린지 (2024) - RTES 연구실 참여 <br>
**공식 홈페이지** :📒 [한국소음진동공학회 AI 챌린지](https://ksnve.notion.site/1-AI-2024-5e0b8972e808498fb29dbb77c42ddf36#9d0b2a7fe13f433a97d1a69470a35b26)<br>
**Project Workspace** :📒 [Team Project Notion](https://www.notion.so/skipper0527/AI-4bd41e7a934b4329960bb453665150ec?pvs=4)<br>
**Data 다운로드 링크** : 📒 [2024 KSNVE AI Challenge data drive](https://drive.google.com/drive/folders/1zDbmSHjl6z7zc7CZE8pB3prgnbfMgLaK)<br>

# 사전 준비사항

## 데이터셋 다운로드

데이터의 용량이 높으므로 git 저장소는 원본 데이터를 포함하지 않는다. 위의 ***Data 다운로드 링크*** 에서 'Track 2 (Crazy data)' 파일을 git clone한 디렉토리에 다운로드한다.

```
result
submission
track2_dataset/
    ㄴ train/
        ㄴ ...
    ㄴ eval/
        ㄴ ...
    ㄴ test/
        ㄴ ...
```

## 의존성 라이브러리 설치

```
pip install -r requirements.txt
```

# train.py: train dataset으로 모델을 훈련하는 코드

딥러닝 모델을 훈련하고자 할 경우 (코드를 수정하여 임의의 전처리 시퀀스, 모델 사용 가능)

```
python train.py
```

# eval.py, test.py: eval, test dataset에 대한 anomaly score 파일 생성 코드

Features를 선택하여 ML 이상탐지 모델을 검증할 경우 (코드를 수정하여 임의의 모델, 특성추출, ML 모델 사용 가능)

```
python eval.py
python test.py
```

# 사전훈련된 모델

사전훈련된 모델의 가중치는 다음 디렉토리에 저장

```
.result/
    ㄴ linearae --> train dataset을 stft2dcae에 넣었을 때 나온 loss값과 y축에서 측정된 train data 대한 TDS(평균, rms, peak)값으로 Linear AutoEncoder를 학습한 결과
    ㄴ stft2dcae --> train dataset의 x,y축 data를 STFT한 것으로 2D Convolutinal AutoEncoder을 학습한 결과
```