import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

train_data_o = pd.read_csv('./train.csv')# survived column is missed

# 先清空數據再分標簽
data = train_data_o[['Age', 'Sex', 'Pclass', 'Survived']].dropna() # 删除有NaN的行

labels = data['Survived'].values
labels = torch.tensor(labels, dtype=torch.float32)
data.drop(['Survived'], axis=1, inplace=True)
print(data.shape, labels.shape)


data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}) # first method

data = data.to_numpy().astype(np.float32) # data = data.values.astype(np.float32)
# standardization
scaler = StandardScaler()
data = scaler.fit_transform(data) # 這個標準化要用numpy類,不能放tensor類
# data = (data - data.mean()) / data.std()


data = torch.from_numpy(data).float()


# 用BCELoss,标签要float
tensor_data = TensorDataset(data, labels)

train_data_size = int(len(tensor_data) * 0.8)
val_data_size = len(tensor_data) - train_data_size
train_dataset, val_dataset = random_split(tensor_data, (train_data_size, val_data_size))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)





class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=3, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.linear3 = nn.Linear(in_features=32, out_features=1)
        self.batchnorm1d1 = nn.BatchNorm1d(64)
        self.batchnorm1d2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.view(-1, 3)
        x = self.linear1(x)
        x = self.batchnorm1d1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.batchnorm1d2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


model = Classifier()

# 二分類用BCELoss
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

for epoch in range(100):
    model.train()
    for i in train_dataloader:
        inputs, labels = i
        inputs = inputs.view(-1, 3)
        labels = labels.view(-1, 1)
        outputs = model(inputs)


        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    total_accuracy = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i in val_dataloader:
            inputs, labels = i
            inputs = inputs.view(-1, 3)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_accuracy += accuracy_score((outputs >= 0.5).int().squeeze(), labels.squeeze()) # 二分類概率大于0.5就是
            total_loss += loss.item()
    print('~'*30 + f'round {epoch+1}, accuracy: {total_accuracy / val_data_size * 100:.2f}%, loss: {total_loss / val_data_size}' + '~'*30)


test_data_o = pd.read_csv('./test.csv')
# test_data = test_data_o[['Age', 'Sex', 'Pclass', 'PassengerId']].dropna()
test_data = test_data_o[['Age', 'Sex', 'Pclass', 'PassengerId']]
test_data = test_data.fillna(0) # 題目不許刪數據
test_ID = test_data.pop('PassengerId')
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})

test_data = test_data.to_numpy().astype(np.float32)
# 测试阶段必须使用相同的scaler 不然这会导致测试数据使用与训练数据不同的分布进行标准化，破坏数据一致性。
test_data = scaler.transform(test_data)
# test_data = (test_data - test_data.mean()) / test_data.std()

test_data = torch.from_numpy(test_data).float()

print(test_data.shape)
test_data = test_data.view(-1, 3)
model.eval()
with torch.no_grad():
    outputs = model(test_data)
    pred = (outputs>=0.5).int().squeeze()
print(pred)

df_test = pd.DataFrame({'PassengerId': test_ID, 'Survived': pred})
print(df_test)

# df_test.to_csv('./submission.csv', index=False)
