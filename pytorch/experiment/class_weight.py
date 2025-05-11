import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# ����ģ�����ݣ�5�����1000��������
X = torch.randn(1000, 10)  # 10������
y = torch.randint(0, 5, (1000,))  # ��ǩ0~4

# �������Ȩ�أ��������2����Ҫ��
class_weights = torch.tensor([0.1, 0.1, 0.5, 0.2, 0.1])

# ����ģ��
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 5)  # �����ڵ���=�����
)

# ������ʧ�������Ż���
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ���� DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ѵ��ѭ��
for epoch in range(10):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")