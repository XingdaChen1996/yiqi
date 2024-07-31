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
import argparse
from yaml_read import read_yaml
import shutil


if __name__ == '__main__':

    # -f config.yaml
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('-f', '--file', type=str, metavar="", required=True,
                        help='path of config.yaml file')

    args = parser.parse_args()
    config_file = read_yaml(args.file)

    if config_file['global']['use_gpu']:
        assert (torch.cuda.is_available()), 'GPU不可用'
        device = torch.device(type="cuda")
    else:
        device = torch.device(type='cpu')

    X = pd.read_excel(config_file['eval']['path'], sheet_name='input', header=0, index_col=0)
    y = pd.read_excel(config_file['eval']['path'], sheet_name='output', header=0, index_col=0)

    with open(os.path.join(config_file['global']['model_save_path'], 'scaler_X.pkl'), "rb") as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(config_file['global']['model_save_path'], 'scaler_y.pkl'), "rb") as f:
        scaler_y = pickle.load(f)

    X = scaler_X.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    model = BPNet(**config_file['global']['model_params']).to(device)
    model.load_state_dict(torch.load(os.path.join(config_file['global']['model_save_path'], 'best_model.pth')))
    model.eval()

    with torch.no_grad():
        pre = model(X)
    if pre.is_cuda:
        pre = pre.cpu()
    pre = scaler_y.inverse_transform(pre.numpy())
    pre = pd.DataFrame(data=pre, index=y.index, columns=y.columns)

    # 计算每行的均方误差
    RMSE_series = np.sqrt(((y - pre) ** 2).mean(axis=1))

    with pd.ExcelWriter(config_file['eval']['output_excel']) as writer:
        pre.to_excel(writer, sheet_name='predict')
        RMSE_series.to_frame().to_excel(writer, sheet_name='rmse')

    print("Mean rmse:", RMSE_series.mean())