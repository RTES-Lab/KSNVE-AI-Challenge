import random
import numpy as np
import pandas as pd
import os

import torch
from torchvision import transforms

import data
import proc
import model.cae as cae
from train import Trainer, multiclass_roc_auc
from sklearn.ensemble import IsolationForest


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loss_from_df(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tf: transforms.Compose,
    loss_fn: torch.nn.Module,
    device: str = "cuda",
):
    losses = []
    m = model.to(device).eval()

    for i in df.index:
        with torch.no_grad():
            x = df.iloc[i]["data"].astype("float32")
            x = tf(x).unsqueeze(0).to(device)
            y = m(x)
            loss = loss_fn(x, y).item()
            losses.append(loss)

    return np.array(losses)


def avg(x):
    return np.mean(x)


def rms(x):
    return np.sqrt(np.mean(x * x))


def pk(x):
    return np.max(np.abs(x))

if __name__ == "__main__":
    set_seed(42)

    print("#1 Generate Dataframe...")
    df_train = data.generate_df("./dataset/train")
    df_val = data.generate_df("./dataset/eval")

    print("#2 Load Models...")
    ########## 불러올 모델의 파라미터 입력 ##########
    # 모델 위치
    models = {
        "cae": torch.load("./result/lstm1dcae/model.pt"),
        "stft_cae": torch.load("./result/stft2dcae/model.pt"),
    }

    # 전처리 시퀀스
    tfs = {
        "cae": transforms.Compose([proc.NpToTensor()]),
        "stft_cae": transforms.Compose([proc.STFT2D(), proc.NpToTensor()]),
    }

    loss = torch.nn.MSELoss()  # 사용할 손실함수
    ##########                  ##########

    print("#3 Fit ML Models...")
    tdss = []

    pred_1dcae = get_loss_from_df(df_train, models["cae"], tfs["cae"], loss)
    pred_stftcae = get_loss_from_df(df_train, models["stft_cae"], tfs["stft_cae"], loss)

    for i in df_train.index:
        x = df_train.iloc[i]["data"].astype("float32")[0]
        y = df_train.iloc[i]["data"].astype("float32")[1]
        tds = [avg(y), rms(y), pk(y)]
        tdss.append(tds)

    features = np.hstack(
        (np.expand_dims(pred_1dcae, 1), np.expand_dims(pred_stftcae, 1), np.array(tdss))
    )

    u = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    X = (features - u) / std

    clf = IsolationForest(random_state=345)
    clf.fit(X)

    print("#4 Evaluate...")
    tdss = []

    pred_1dcae = get_loss_from_df(df_val, models["cae"], tfs["cae"], loss)
    pred_stftcae = get_loss_from_df(df_val, models["stft_cae"], tfs["stft_cae"], loss)

    for i in df_val.index:
        x = df_val.iloc[i]["data"].astype("float32")[0]
        y = df_val.iloc[i]["data"].astype("float32")[1]
        tds = [avg(y), rms(y), pk(y)]
        tdss.append(tds)

    features = np.hstack(
        (np.expand_dims(pred_1dcae, 1), np.expand_dims(pred_stftcae, 1), np.array(tdss))
    )

    X = (features - u) / std

    gt = df_val["label"].values.copy()
    yy = clf.decision_function(X)
    yy = (yy - 1) * -1

    class_aucs, total_auc = multiclass_roc_auc(np.array(gt), np.array(yy))

    for i, class_auc in enumerate(class_aucs):
        print(f"CLASS {i}, ROC-AUC {class_auc}")
    print(f"TOTAL, ROC-AUC {total_auc}")
