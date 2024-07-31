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
from pso import pso


if __name__ == '__main__':

    # -f config.yaml
    parser = argparse.ArgumentParser(description="reverse")
    parser.add_argument('-f', '--file', type=str, metavar="", required=True,
                        help='path of config.yaml file')

    args = parser.parse_args()
    config_file = read_yaml(args.file)

    if config_file['global']['use_gpu']:
        assert (torch.cuda.is_available()), 'GPU不可用'
        device = torch.device(type="cuda")
    else:
        device = torch.device(type='cpu')

    X = pd.read_excel(config_file['reverse']['path'], sheet_name='input', header=0, index_col=0)
    y = pd.read_excel(config_file['reverse']['path'], sheet_name='output', header=0, index_col=0)

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

    with pd.ExcelWriter(config_file['reverse']['output_excel']) as writer:
        pre.to_excel(writer, sheet_name='423453')


    def fun1(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def fun2(x):
        n = len(x)
        return 10 * n + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

    bounds = np.array([(5, 50), (-10, 10), (-10, 10)])
    num_particles = 100
    max_iterations = 100
    image_path = '1.png'
    solution, fitness = pso(obj_fun=fun1,
                            bounds=bounds,
                            num_particles=num_particles,
                            max_iterations=max_iterations,
                            image_path=image_path)

