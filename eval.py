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

    file_names = df["filename"].values.tolist()
    with torch.no_grad():
        for i, k in enumerate(loader):
            x, y = k

            x = x.to("cuda")
            y = y.to("cuda")

            yhat = model(x)
            loss = loss_fn()(yhat, x)

            log["File"].append(file_names[i])
            log["Score"].append(loss.item())

    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(result_dir, exist_ok=True)
    pd.DataFrame(log).to_csv(f"{result_dir}/{filename}", index=False)
    
def extract_key(file_name):
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    parts = base_name.split('_')
    
    # 접두사 (vibration_ball, vibration_inner 등)
    prefix = "_".join(parts[:2])
    
    # 숫자 부분
    first_number = int(parts[2])
    second_number = int(parts[3])
    
    return (prefix, first_number, second_number)

def generate_df2(root: str) -> pd.DataFrame:
    file_list = os.listdir(root)
    file_list = sorted(file_list, key=extract_key)

    df = {"filename": [], "data": [], "label": []}

    for filename in file_list:
        data = pd.read_csv(f"{root}/{filename}")
        x = data["bearingB_x"].values.ravel()
        y = data["bearingB_y"].values.ravel()
        cat_data = np.vstack((x, y))
        df["filename"].append(filename)
        df["data"].append(cat_data)
        df["label"].append(0)

    df = pd.DataFrame(df)
    return df

if __name__ == "__main__":
    print("#1 Generate Dataframe...")
    df_val = generate_df2("./track2_dataset/eval")
    
    print("#2 Calculate loss score and save to CSV for validation set...")
    calculate_score_and_save_csv(
        result_dir="./submission",
        loss_fn=torch.nn.MSELoss,
        df=df_val,
        filename="eval_score.csv",
        batch_size=1,
        n_workers=0
    )