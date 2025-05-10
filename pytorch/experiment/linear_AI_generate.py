import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 生成数据
x = torch.linspace(0, 1, 100).unsqueeze(1)
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + torch.randn(x.size()) * 0.1

# 定义模型
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练模型
epochs = 100
for epoch in range(epochs):
    predictions = model(x)
    loss = criterion(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 获取训练后的参数
w_trained = model.weight.data.item()
b_trained = model.bias.data.item()
print(f'训练后的权重 w: {w_trained:.4f}, 偏置 b: {b_trained:.4f}')

# 预测并可视化
with torch.no_grad():
    y_pred = model(x).numpy()

plt.scatter(x.numpy(), y.numpy(), label='帶噪聲的數據')
plt.plot(x.numpy(), (true_w * x + true_b).numpy(), 'r', label='直實直線')
plt.plot(x.numpy(), y_pred, 'g--', label='預測直線')
plt.legend()
plt.show()