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

# reproducibility를 위해

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

def train_model(result_dir,
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
                df_val,
                seed,
                verbose=True):
    set_seed(seed)
    train_loader = data.get_dataloader(df_train, tf, None, True, batch_size, n_workers)
    val_loader = data.get_dataloader(df_val, tf, None, False, 1, n_workers)

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
    log = trainer.val()
    trainer.save(result_dir)

    return trainer

if __name__ == "__main__":
    print("#1 Generate Dataframe...")
    df_train = data.generate_df("./dataset/train")
    df_val = data.generate_df("./dataset/eval")

    print("#2 Train LSTM CAE...")
    train_model(
        result_dir="./result/lstm1dcae",
        model=cae.LSTM1DCAE,
        model_kwargs={"in_channels": 2},
        optim=torch.optim.Adam,
        optim_kwargs={"lr": 0.004},
        loss=torch.nn.MSELoss,
        loss_kwargs=None,
        tf=transforms.Compose([proc.NpToTensor(),]),
        batch_size=16,
        n_epochs=400,
        n_workers=0,
        df_train=df_train,
        df_val=df_val,
        seed=42,
        verbose=True
    )

    print("#3 Train STFT CAE...")
    train_model(
        result_dir="./result/stft2dcae",
        model=cae.Simple2DSTFTCAE,
        model_kwargs={"in_channels": 2},
        optim=torch.optim.Adam,
        optim_kwargs={"lr": 0.004},
        loss=torch.nn.MSELoss,
        loss_kwargs=None,
        tf=transforms.Compose([proc.STFT2D(),proc.NpToTensor(),]),
        batch_size=16,
        n_epochs=400,
        n_workers=0,
        df_train=df_train,
        df_val=df_val,
        seed=42,
        verbose=True
    )