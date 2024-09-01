from typing import Dict, Any
import os
import pandas as pd
import numpy as np

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import roc_auc_score

class Trainer:
    def __init__(self, model: torch.nn.Module,
                 model_kwargs: Dict,
                 optimizer: torch.optim.Optimizer,
                 optimizer_kwargs: Dict,
                 loss: torch.nn.modules.loss._Loss,
                 loss_kwargs: Dict,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str) -> None:
        if model_kwargs is not None:
            self.model = model(**model_kwargs)
        else:
            self.model = model()
        
        if optimizer_kwargs is not None:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = optimizer(self.model.parameters())
        
        if loss_kwargs is not None:
            self.loss_fn = loss(**loss_kwargs)
        else:
            self.loss_fn = loss()
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.model = self.model.to(device)

        self.train_log = None
        self.val_log = None
    
    def train_one_epoch_(self):
        losses = []

        for _, data in enumerate(self.train_loader):
            x, y = data
            self.optimizer.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)

            yhat = self.model(x)

            loss = self.loss_fn(yhat, x)
            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())
        
        avg_loss = np.mean(np.array(losses))

        return avg_loss
    
    def train(self, n_epochs, verbose=False):
        self.train_log = {
            "loss": [],
            "epoch": []
        }
        self.model = self.model.train()
        for i in range(n_epochs):
            loss = self.train_one_epoch_()
            if verbose:
                print(f"EPOCH [{i+1}/{n_epochs}]: train_loss = {loss}")
            self.train_log["loss"].append(loss)
            self.train_log["epoch"].append(i+1)
        
        return self.train_log
    
    def val(self):
        self.val_log = {
            "pred": [],
            "gt": []
        }

        self.model = self.model.eval()

        with torch.no_grad():
            for _, data in enumerate(self.val_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)

                yhat = self.model(x)
                loss = self.loss_fn(yhat, x)

                self.val_log["pred"].append(loss.item())
                self.val_log["gt"].append(y.item())
        
        return self.val_log
    
    def save(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.model, f"{dir}/model.pt")


def multiclass_roc_auc(gt, pred):
    n_cls = np.max(gt)
    class_score = []

    for i in range(1, n_cls+1):
        idx = (gt == 0) | (gt == i)
        pred_class = pred[idx]
        gt_class = gt[idx]
        gt_class[gt_class != 0] = 1
        
        roc_auc = roc_auc_score(gt_class, pred_class)

        class_score.append(roc_auc)

    pred_class = pred
    gt_class = gt
    gt_class[gt_class != 0] = 1

    total_score = roc_auc_score(gt_class, pred_class)

    return class_score, total_score
