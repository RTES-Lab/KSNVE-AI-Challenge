import random
import numpy as np
import pandas as pd
import os

import torch
from torchvision import transforms

import data
import proc
import model.cae as cae
from trainer import Trainer, multiclass_roc_auc

# reproducibility를 위해 seed 초기 설정

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

# STFT2dCAE 학습 함수
def train_stft(result_dir,
                model,
                model_kwargs,
                optim,
                optim_kwargs,
                loss,
                loss_kwargs,
                tf,
                batch_size,
                n_epochs,
                n_workers,
                df_train,
                seed,
                verbose=True):
    set_seed(seed)
    train_loader = data.get_dataloader(df_train, tf, None, True, batch_size, n_workers)

    trainer = Trainer(
    model,
    model_kwargs,
    optim,
    optim_kwargs,
    loss,
    loss_kwargs,
    train_loader,
    None,
    "cuda",
    )

    trainer.train(n_epochs, verbose)
    trainer.save(result_dir)

    return trainer

# LinearAE 학습 함수
def train_linear(result_dir,
                model,
                model_kwargs,
                optim,
                optim_kwargs,
                loss,
                loss_kwargs,
                batch_size,
                n_epochs,
                n_workers,
                df_train,
                df_val,
                seed,
                verbose=True):
    set_seed(seed)

    # 1. train_loader 생성
    stft_cae = torch.load("./result/stft2dcae/model.pt")
    stft_tfs = transforms.Compose([proc.STFT2D(), proc.NpToTensor()])

    train_features, u, std = data.generate_features(df_train, stft_cae, stft_tfs, loss)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features).float(), torch.from_numpy(df_train["label"].values).int())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)


    # 2. val_loader 생성
    val_features, _, _ = data.generate_features(df_val, stft_cae, stft_tfs, loss, u, std)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(df_val["label"].values).int())
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=n_workers, shuffle=False)


    # 3. 학습 및 테스트
    trainer = Trainer(
    model,
    model_kwargs,
    optim,
    optim_kwargs,
    loss,
    loss_kwargs,
    train_loader,
    val_loader,
    "cuda",
    )

    trainer.train(n_epochs, verbose)
    trainer.save(result_dir)

    return trainer

if __name__ == "__main__":
    print("#1 Generate Dataframe...")
    df_train = data.generate_df("./track2_dataset/train")
    df_val = data.generate_df("./track2_dataset/eval")

    print("#2 Train STFT CAE...")
    train_stft(
        result_dir="./result/stft2dcae",
        model=cae.Simple2DSTFTCAE,
        model_kwargs={"in_channels": 2},
        optim=torch.optim.Adam,
        optim_kwargs={"lr": 0.004},
        loss=torch.nn.MSELoss,
        loss_kwargs=None,
        tf=transforms.Compose([proc.STFT2D(),proc.NpToTensor(),]),
        batch_size=32,
        n_epochs=500,
        n_workers=0,
        df_train=df_train,
        seed=42,
        verbose=True
    )

    print("#3 Train Linear AE...")
    train_linear(
        result_dir="./result/linearae",
        model=cae.LinearAE,
        model_kwargs={"in_channels": 4},
        optim=torch.optim.Adam,
        optim_kwargs={"lr": 0.004},
        loss=torch.nn.MSELoss,
        loss_kwargs=None,
        batch_size=32,
        n_epochs=200,
        n_workers=0,
        df_train=df_train,
        df_val=df_val,
        seed=42,
        verbose=True
    )
