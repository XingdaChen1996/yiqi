import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import dill as pickle
from BPNet import BPNet


if __name__ == '__main__':
    path = "舱室声压级20240729_eval.xlsx"
    scaler_save_dir = 'model'
    model_save_path = 'model'
    model_params = {'input_dim': 16, 'output_dim': 794}
    predict_dir = 'eval.xlsx'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = pd.read_excel(path, sheet_name='input', header=0, index_col=0)
    y = pd.read_excel(path, sheet_name='output', header=0, index_col=0)

    with open(os.path.join(model_save_path, 'scaler_X.pkl'), "rb") as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(model_save_path, 'scaler_y.pkl'), "rb") as f:
        scaler_y = pickle.load(f)

    X = scaler_X.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    model = BPNet(**model_params).to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model.pth')))
    model.eval()

    with torch.no_grad():
        pre = model(X)
    pre = scaler_y.inverse_transform(pre.numpy())
    pre = pd.DataFrame(data=pre, index=y.index, columns=y.columns)

    # 计算每行的均方误差
    RMSE_series = np.sqrt(((y - pre) ** 2).mean(axis=1))

    with pd.ExcelWriter(predict_dir) as writer:
        pre.to_excel(writer, sheet_name='predict')
        RMSE_series.to_frame().to_excel(writer, sheet_name='rmse')

    print("Mean rmse:", RMSE_series.mean())