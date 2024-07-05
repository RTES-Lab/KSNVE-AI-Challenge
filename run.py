import random
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

import data
import proc
import model.cae as cae
from train import Trainer, multiclass_roc_auc

# reproducibility를 위해

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

print("#1 Generate Dataframe...")

df_train = data.generate_df("./dataset/train")
df_val = data.generate_df("./dataset/eval")

########## 훈련에 필요한 파라미터 입력 ##########
result_dir = "./logs/test"  # 훈련 결과를 저장할 위치

# 데이터 전처리 시퀀스
tf = transforms.Compose(
    [
        proc.Polar(),
        proc.NpToTensor(),
    ]
)

batch_size = 16  # 배치 크기
n_epochs = 400  # 에폭 수
n_workers = 8  # 데이터 로딩에 사용할 코어수

model = cae.LSTM1DCAE  # 사용할 모델
model_kwargs = None  # 사용할 모델에 들어갈 인자

optim = torch.optim.Adam  # 사용할 최적화 알고리즘
optim_kwargs = {"lr": 0.004}  # 사용할 최적화 알고리즘에 들어갈 인자

loss = torch.nn.MSELoss  # 사용할 손실함수
loss_kwargs = None  # 사용할 손실함수에 들어갈 인자

use_md = True  # 마할라노비스 거리 사용할건지?
##########                  ##########

print("#2 Generate DataLoader...")

train_loader = data.get_dataloader(df_train, tf, None, True, batch_size, n_workers)

val_loader = data.get_dataloader(df_val, tf, None, False, 1, n_workers)

print("#3 Train Model...")

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

trainer.train(n_epochs, True)
log = trainer.val()
trainer.save(result_dir)

# 마할라노비스 거리 기반으로 prediction 다시 계산

if use_md:
    log = {"pred": [], "gt": []}

    m = torch.load(f"{result_dir}/model.pt")

    losses = []

    for i in df_train.index:
        with torch.no_grad():
            x = df_train.iloc[i]["data"].astype("float32")
            x = tf(x).unsqueeze(0).to("cuda")
            y = m(x)
            loss = trainer.loss_fn(x, y).item()
            losses.append(loss)

    mh_u = np.mean(np.array(losses))
    mh_std = np.std(np.array(losses))

    for i in df_val.index:
        with torch.no_grad():
            x = df_val.iloc[i]["data"].astype("float32")
            x = tf(x).unsqueeze(0).to("cuda")
            y = m(x)
            loss = trainer.loss_fn(x, y).item()
            mhd = abs(loss - mh_u) / mh_std
            log["pred"].append(mhd)
            log["gt"].append(df_val.iloc[i]["label"])

    trainer.val_log = log
    trainer.save(result_dir)

print("#4 Evaluate Model...")

class_aucs, total_auc = multiclass_roc_auc(np.array(log["gt"]), np.array(log["pred"]))

for i, class_auc in enumerate(class_aucs):
    print(f"CLASS {i}, ROC-AUC {class_auc}")
print(f"TOTAL, ROC-AUC {total_auc}")
