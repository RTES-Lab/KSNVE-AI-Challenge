import os

import pandas as pd
import numpy as np
import torch
import proc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from typing import Tuple


def generate_df(root: str) -> pd.DataFrame:
    file_list = os.listdir(root)
    file_list.sort()
    fault_map = {"normal": "H", "inner": "IR", "outer": "OR", "ball": "B"}

    label_map = {"normal": 0, "inner": 1, "outer": 2, "ball": 3}
    anomaly_map = {"normal": 0, "inner": 1, "outer": 1, "ball": 1}

    df = {"data": [], "fault_type": [], "label": [], "anomaly": []}

    for filename in file_list:
        data = pd.read_csv(f"{root}/{filename}")
        x = data["bearingB_x"].values.ravel()
        y = data["bearingB_y"].values.ravel()
        cat_data = np.vstack((x, y))
        df["data"].append(cat_data)

        fault = filename.split("_")[1]

        df["fault_type"].append(fault_map[fault])
        df["label"].append(label_map[fault])
        df["anomaly"].append(anomaly_map[fault])

    df = pd.DataFrame(df)
    return df


class PandasDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.transforms.Compose = None,
        target_transform: transforms.transforms.Compose = None,
    ) -> None:
        self.df = df.reset_index()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        x = self.df.iloc[idx]["data"].astype("float32")
        t = self.df.iloc[idx]["label"].astype("int64")

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            t = self.target_transform(t)

        return x, t


def get_dataloader(
    df: pd.DataFrame,
    transform_data: transforms.transforms.Compose,
    transform_label: transforms.transforms.Compose,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = PandasDataset(df, transform_data, transform_label)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

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

def generate_features(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tf: transforms.Compose,
    loss_fn: torch.nn.Module,
    u = None,
    std = None
):
    # STFT CAE loss 계산
    pred_stftcae = get_loss_from_df(df, model, tf, loss_fn())

    # TDS features 계산
    tdss = []
    for i in df.index:
        y = df.iloc[i]["data"].astype("float32")[1]
        tds = [proc.avg(y), proc.rms(y), proc.pk(y)]
        tdss.append(tds)

    #  STFT loss와 TDS features 결합
    features = np.hstack((np.expand_dims(pred_stftcae, 1), np.array(tdss)))

    # 결합한 feautres normalize (모델을 처음 훈련하는 경우, 이후 validation, test 단계에서 사용하기 위해 u, std를 저장)
    if u is None or std is None:
        u = np.mean(features, axis=0)
        std = np.std(features, axis=0)

        save_dir = "./output"
        save_path = os.path.join(save_dir, "parameters.npz")
        os.makedirs(save_dir, exist_ok=True)
        
        np.savez(save_path, u=u, std=std)

    features = (features - u) / std
    return features, u, std