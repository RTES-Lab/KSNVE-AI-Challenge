# KSNVE-AI-Challenge
제 1회 한국소음진동공학회 AI 챌린지 (2024) - RTES 연구실 참여 <br>
**공식 홈페이지** :📒 [한국소음진동공학회 AI 챌린지](https://ksnve.notion.site/1-AI-2024-5e0b8972e808498fb29dbb77c42ddf36#9d0b2a7fe13f433a97d1a69470a35b26)<br>
**Project Workspace** :📒 [Team Project Notion](https://www.notion.so/skipper0527/AI-4bd41e7a934b4329960bb453665150ec?pvs=4)<br>
**Data 다운로드 링크** : 📒 [2024 KSNVE AI Challenge data drive](https://drive.google.com/drive/folders/1zDbmSHjl6z7zc7CZE8pB3prgnbfMgLaK)<br>
<br>

# 시스템 아키텍처
![모델 구조](https://github.com/RTES-Lab/KSNVE-AI-Challenge/blob/final/model_architecture.png)

제안된 STFT-TDS Fusion AutoEncoder (STFT-TDS FAE) 모델은 구름 베어링의 이상 탐지를 위해 설계된 혁신적인 아키텍처로, 세 가지 핵심 컴포넌트로 구성된다: Short-Time Fourier Transform (STFT) 기반 2D Convolutional AutoEncoder (CAE), Time Domain Statistics (TDS) 추출기, 그리고 Linear AutoEncoder (AE)이다. 본 모델은 베어링의 x축과 y축 방향에서 가속도계로 측정된 raw 진동 신호를 입력 데이터로 활용한다.

 첫 번째 컴포넌트인 STFT 2D CAE는 입력 신호의 시간-주파수 표현을 학습한다. 구체적으로, x축과 y축의 raw 데이터는 STFT를 통해 2차원 시간-주파수 행렬로 변환되며, 이는 2D CAE의 입력으로 사용된다. 2D CAE는 이 변환된 데이터로부터 주파수 도메인의 feature를 추출하고, reconstruction 과정을 통해 $L_{STFT}$라는 손실 값을 생성한다. 

 두 번째 컴포넌트인 TDS 추출기는 y축 raw 데이터의 시간 도메인 특성을 포착한다. 이 과정에서 세 가지 핵심 통계량이 계산된다: 평균 μ(y), 최대값 pk(y), 제곱평균제곱근 rms(y). 이들 통계량은 TDS(y)로 표현되며, 시간 영역에서의 신호 특성을 대표하는 중요한 지표이다.

 마지막으로, 두 컴포넌트의 출력을 통합하는 Linear AE는 총 네 개의 특징($L_{STFT}$와 TDS의 세 가지 통계량)을 입력으로 받아 처리한다. 이 과정을 통해 Linear AE는 주파수 및 시간 도메인의 정보를 효과적으로 융합하여 최종적인 이상 점수인 $L_{Total}$을 산출한다. $L_{Total}$은 입력 데이터의 종합적인 이상 정도를 나타내는 anomaly score로, 그 값이 높을수록 해당 데이터가 이상 상태일 가능성이 높음을 의미한다.
<br><br>
# 사전 준비사항

## 데이터셋 다운로드

데이터의 용량이 높으므로 git 저장소는 원본 데이터를 포함하지 않는다. 위의 **Data 다운로드 링크** 에서 **'Track 2 (Crazy data)'** 파일을 git clone한 디렉토리에 다운로드한다.

```
.result
.submission
.track2_dataset/
    ㄴ train/
        ㄴ ...
    ㄴ eval/
        ㄴ ...
    ㄴ test/
        ㄴ ...
```

## 의존성 라이브러리 설치
코드 실행에 필요한 라이브러리를 설치해야한다. (Cuda 11.7 버전 기준)

```
pip install -r requirements.txt
```
<br>

# 디렉토리 구성 요소
## train.py
train dataset으로 모델을 훈련하는 코드

```
python train.py
```

## eval.py / test.py
eval, test dataset에 대한 anomaly score 파일을 생성하는 코드

```
python eval.py
python test.py
```

## data.py / proc.py / trainer.py
- **data.py**: 학습 데이터셋을 만드는 함수 및 클래스가 있다.
- **proc.py**: 데이터 전처리를 수행하는 함수 및 클래스가 있다.
- **trainer.py**: 모델을 학습하고 평가하기 위한 함수 및 클래스가 있다.
  
## model / output / result / submission
- **model**: 2D STFT Convolutional AutoEncoder, Linear AutoEncoder model 코드가 있다.
- **output**: Linear AutoEncoder의 학습 및 추론에 사용할 데이터들의 평균과 분산이 저장되어 있다.
- **result**: 2D STFT Convolutional AutoEncoder, Linear AutoEncoder의 학습된 가중치와 학습 및 검증 log가 있다.
```
.result/
    ㄴ linearae --> train dataset의 x, y축 data를 stft2dcae에 넣었을 때 나온 loss값과 y축에서 측정된 train data 대한 TDS(평균, 최대값, 제곱평균제곱근)값으로 Linear AutoEncoder를 학습한 결과
    ㄴ stft2dcae --> train dataset의 x, y축 data를 STFT한 것으로 2D Convolutinal AutoEncoder을 학습한 결과
```
- **submission**: eval, test data의 anomaly score 파일이 있다.
