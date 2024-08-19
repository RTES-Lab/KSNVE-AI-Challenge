import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import data
import proc
import model.cae as cae
import os

def calculate_score_and_save_csv(result_dir, loss_fn, df, filename, batch_size=1, n_workers=0):
    """Calculate loss for a dataset and save to a CSV file."""
    
    # 모델 로드
    model = torch.load(f"./result/linearae/model.pt")
    model = model.to("cuda")
    model.eval()
    
    stft_cae = torch.load(f"./result/stft2dcae/model.pt")
    stft_tfs = transforms.Compose([proc.STFT2D(), proc.NpToTensor()])
    
    # 학습 데이터의 u, std를 load
    d = np.load("./output/parameters.npz")
    u= d['u']
    std= d['std']

    # 특징 추출 및 손실 계산
    features, _, _ = data.generate_features(df, stft_cae, stft_tfs, loss_fn, u, std)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(df["label"].values).int())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)

    log = {
            "File": [],
            "Score": []
            
    }
    with torch.no_grad():
        for i, k in enumerate(loader):
            x, y = k

            x = x.to("cuda")
            y = y.to("cuda")

            yhat = model(x)
            loss = loss_fn()(yhat, x)

            file_number = f"{i:04d}"
            log["File"].append(file_number)
            log["Score"].append(loss.item())

    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(result_dir, exist_ok=True)
    pd.DataFrame(log).to_csv(f"{result_dir}/{filename})", index=False)
    

def generate_df2(root: str) -> pd.DataFrame:
    file_list = os.listdir(root)
    file_list.sort()

    df = {"data": [], "label": []}

    for filename in file_list:
        data = pd.read_csv(f"{root}/{filename}")
        x = data["bearingB_x"].values.ravel()
        y = data["bearingB_y"].values.ravel()
        cat_data = np.vstack((x, y))
        df["data"].append(cat_data)
        df["label"].append(0)

    df = pd.DataFrame(df)
    return df

if __name__ == "__main__":
    print("#1 Generate Dataframe...")
    df_val = generate_df2("./track2_dataset/test")
    


    print("#2 Calculate loss score and save to CSV for test set...")
    calculate_score_and_save_csv(
        result_dir="./submission",
        loss_fn=torch.nn.MSELoss,
        df=df_val,
        filename="validation_results.csv",
        batch_size=1,
        n_workers=0
    )