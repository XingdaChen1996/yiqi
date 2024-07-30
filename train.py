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
    path = "舱室声压级20240729_train.xlsx"
    val_size = 0.1
    num_epochs = 100
    random_state = 0
    batch_size = 32
    log_dir = 'log'
    scaler_save_dir = 'model'
    model_save_path = 'model'
    model_params = {'input_dim': 16, 'output_dim': 794}

    # read and split data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = pd.read_excel(path, sheet_name='input', header=0, index_col=0)
    y = pd.read_excel(path, sheet_name='output', header=0, index_col=0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = BPNet(**model_params).to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构
    print(model)
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
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

        with open('log.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

    writer.close()

    with open(os.path.join(model_save_path, 'scaler_X.pkl'), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(model_save_path, 'scaler_y.pkl'), "wb") as f:
        pickle.dump(scaler_y, f)

    print('Training complete')
