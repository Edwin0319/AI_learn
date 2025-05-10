import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_X = np.array([[2.25, 0.85, 0.2, 0.45]])
data_Y = np.array([[0.35, 0.4, 0.61, 0.5]])
data_X = data_X.reshape(-1, 1)
data_Y = data_Y.reshape(-1, 1)
data_X = torch.tensor(data_X, dtype=torch.float32)
data_Y = torch.tensor(data_Y, dtype=torch.float32)
print(data_X)
print(data_Y)

# 简单曲线（如二次函数）：1~2个隐藏层，每层64个神经元。
# 复杂曲线（如高频正弦波）：增加层数（如3~4层），或使用更多神经元（如256个）。

class Lin(nn.Module):
    def __init__(self):
        super(Lin, self).__init__()
        self.linear1 = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        return x

class Curve(nn.Module):
    def __init__(self):
        super(Curve, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=1, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


model = Lin()
# model = Curve()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.MSELoss()
for epoch in range(1, 1001):
    prediction = model(data_X)

    loss = criterion(prediction, data_Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # print(f'~~~~~~~~~~{epoch}~~~~~~~~~~, Loss: {loss.item():.4f}')
        pass

test_X = torch.linspace(0, 3, 100)
test_X = test_X.reshape(-1, 1)
test_Y = model(test_X)

test_X = test_X.detach().numpy()
test_Y = test_Y.detach().numpy()

plt.scatter(data_X, data_Y, color='blue', label='real_data')
plt.plot(test_X, test_Y, color='red', label='prediction')
plt.legend()
plt.show()