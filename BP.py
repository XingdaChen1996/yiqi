"""
@Time    : 2024-07-29
@Author  : xingdachen
@Email   : chenxingda@iat-center.com
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# 定义神经网络
class BPNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=1)
        return x


if __name__ == '__main__':
    path = "舱室声压级20240729.xlsx"

    epoch = 100
    X = pd.read_excel(path, sheet_name='input')
    y = pd.read_excel(path, sheet_name='output')
    # 实例化模型
    model = BPNet(X.shape[1], y.shape[1])

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构
    print(model)

    # 训练模型
    # 假设你有训练数据 x_train (形状为 [样本数量, 16]) 和 y_train (形状为 [样本数量])
    # 训练循环:
    for i in range(epoch):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{i+1}/10], Loss: {loss.item():.4f}')

    print(1)