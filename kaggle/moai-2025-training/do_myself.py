import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, random_split, DataLoader
import torchvision
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# TODO: 讀取數據集，'train-images.pt' 和 'train-labels.csv'
images_raw = torch.load("/kaggle/input/moai-2025-training/train_images.pt", weights_only=False) #
labels_raw = pd.read_csv("/kaggle/input/moai-2025-training/train_labels.csv")

# TODO: 歸一化數據集並轉換為 torch.Tensor
images = (images_raw.float() - images_raw.float().mean()) / images_raw.float().std()
labels = torch.tensor(labels_raw['label'].values) #

# TODO: 創建數據集，並按照 8:2 劃分成訓練集和驗證集
dataset = TensorDataset(images, labels)
train_data_size = int(len(dataset) * 0.8) #
test_data_size = len(dataset) - train_data_size
print(train_data_size, test_data_size)
train_dataset, val_dataset = random_split(dataset, (train_data_size, test_data_size))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

for i, j in train_loader:
    pass
print(i.shape, j.shape)
for i, j in val_loader:
    pass
print(i.shape, j.shape)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # TODO: 定義網絡層
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # TODO: 定義前向傳播
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


from sklearn.metrics import accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# TODO: 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(5):
    # TODO: 訓練循環
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        optimizer.zero_grad()
        train_loss = criterion(output, labels.long())
        train_loss.backward()
        optimizer.step()

        # 不確定最后一批數據是否夠64整
        # train_acc = (output.argmax(1) == labels).sum() / 64
        train_acc = accuracy_score(output.argmax(1), labels)
        # 以下是記錄損失函數和準確率的代碼，不用修改
        if batch_idx % 50 == 0:
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(
                f"Epoch {epoch + 1}, Batch {batch_idx}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")
    print(images.shape)
    # TODO: 驗證循環
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            # 不確定最后一批數據是否夠64整
            # val_acc = (output.argmax(1) == labels).sum() / 64
            val_acc = accuracy_score(output.argmax(1), labels)
            val_loss = criterion(output, labels)
    print(images.shape)
    # 以下是記錄損失函數和準確率的代碼，不用修改
    val_accs.append(val_acc)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}")





import torch
import pandas as pd

test_images = torch.load('/kaggle/input/moai-2025-training/test_images.pt', weights_only=True)
# TODO: 按照前面的方法歸一化
test_images = (test_images.float() - test_images.float().mean()) / test_images.float().std()

model.eval()
with torch.no_grad():
    test_images = test_images.to(device)
    outputs = model(test_images)
    predictions = outputs.argmax(dim=1)

df_test = pd.DataFrame({"label": predictions.cpu().numpy()})
df_test.to_csv("submission.csv", index_label="id")
