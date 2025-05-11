import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# 生成模拟数据（5个类别，1000个样本）
X = torch.randn(1000, 10)  # 10个特征
y = torch.randint(0, 5, (1000,))  # 标签0~4

# 假设类别权重（例如类别2最重要）
class_weights = torch.tensor([0.1, 0.1, 0.5, 0.2, 0.1])

# 定义模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 5)  # 输出层节点数=类别数
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建 DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for epoch in range(10):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")