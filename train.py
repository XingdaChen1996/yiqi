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


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder '{folder_path}' and its contents have been deleted.")
    else:
        print(f"The folder '{folder_path}' does not exist.")


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"The folder '{folder_path}' has been created.")
    else:
        print(f"The folder '{folder_path}' already exists.")


if __name__ == '__main__':

    # -f config.yaml
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('-f', '--file', type=str, metavar="", required=True,
                        help='path of config.yaml file')

    args = parser.parse_args()
    config_file = read_yaml(args.file)

    delete_folder(config_file['global']['model_save_path'])
    delete_folder(config_file['global']['log_dir'])

    create_folder(config_file['global']['model_save_path'])
    create_folder(config_file['global']['log_dir'])

    # read and split data
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config_file['global']['use_gpu']:
        assert (torch.cuda.is_available()), 'GPU不可用'
        device = torch.device(type="cuda")
    else:
        device = torch.device(type='cpu')

    X = pd.read_excel(config_file['train']['path'], sheet_name='input', header=0, index_col=0)
    y = pd.read_excel(config_file['train']['path'], sheet_name='output', header=0, index_col=0)
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=config_file['train']['val_size'],
                                                      random_state=config_file['train']['random_state'])

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(X)
    scaler_y.fit(y)

    X_train = scaler_X.transform(X_train)
    X_val = scaler_X.transform(X_val)
    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config_file['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config_file['train']['batch_size'], shuffle=False)

    # 实例化模型
    model = BPNet(**config_file['global']['model_params']).to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构
    print(model)
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=config_file['global']['log_dir'])
    best_val_loss = float('inf')

    for epoch in range(config_file['train']['num_epochs']):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        # Log the training and validation loss to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        with open(os.path.join(config_file['global']['log_dir'], 'log.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config_file['global']['model_save_path'], 'best_model.pth'))

    writer.close()

    with open(os.path.join(config_file['global']['model_save_path'], 'scaler_X.pkl'), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(config_file['global']['model_save_path'], 'scaler_y.pkl'), "wb") as f:
        pickle.dump(scaler_y, f)

    print('Training complete')
