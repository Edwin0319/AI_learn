"""
泰坦尼克号生存预测完整代码
包含数据预处理、模型训练、验证和预测全流程
"""

# 导入库
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split

# 设置随机种子保证可复现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------- 数据预处理函数 -----------------
def preprocess_data(df, imputer=None, scaler=None, is_train=True):
    """
    数据预处理管道
    Args:
        df: 原始数据框
        imputer: 预训练的缺失值填充器（测试集时传入）
        scaler: 预训练的数据标准化器（测试集时传入）
        is_train: 是否为训练模式
    Returns:
        处理后的特征张量，标签张量（测试集为None），及预处理器对象
    """
    try:
        # 特征选择与复制（避免修改原始数据）
        data = df[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare']].copy()

        # 第一步：转换类别型特征
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}).astype(float)

        # 第二步：处理缺失值（训练模式拟合新转换器，测试模式使用现有转换器）
        if imputer is None:
            imputer = SimpleImputer(strategy='median')  # 中位数填充策略

        if is_train:
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)

        # 第三步：标准化处理
        if scaler is None:
            scaler = StandardScaler()

        if is_train:
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        else:
            data = pd.DataFrame(scaler.transform(data), columns=data.columns)

        # 转换为PyTorch张量
        data_tensor = torch.tensor(data.values, dtype=torch.float32)

        # 返回结果
        if is_train:
            labels = df['Survived'].values
            labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
            return data_tensor, labels_tensor, imputer, scaler
        else:
            return data_tensor, imputer, scaler

    except Exception as e:
        print(f"数据预处理错误: {str(e)}")
        raise


# ----------------- 神经网络模型定义 -----------------
class SurvivalClassifier(nn.Module):
    """生存预测神经网络模型"""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ----------------- 训练流程函数 -----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100):
    """
    模型训练与验证流程
    Args:
        model: 初始化的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
    Returns:
        训练好的最佳模型
    """
    best_val_acc = 0
    current_lr = optimizer.param_groups[0]['lr']

    try:
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    outputs = model(X)
                    val_loss += criterion(outputs, y).item()
                    preds = (outputs >= 0.5).float()
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            # 计算指标
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total

            # 更新学习率
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']

            # 学习率变化检测
            if new_lr != current_lr:
                print(f"\n学习率从 {current_lr:.2e} 调整到 {new_lr:.2e}")
                current_lr = new_lr

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # torch.save(model.state_dict(), 'best_model.pth')
                pass

            # 打印训练信息
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            print(f"验证准确率: {val_acc:.2f}% | 当前学习率: {current_lr:.2e}")
            print("-" * 60)

        print(f"\n最佳验证准确率: {best_val_acc:.2f}%")
        return model

    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise


# ----------------- 主程序流程 -----------------
if __name__ == "__main__":
    try:
        # 加载数据
        train_df = pd.read_csv('./train.csv')
        test_df = pd.read_csv('./test.csv')

        # 数据预处理
        print("正在进行数据预处理...")
        train_data, train_labels, imputer, scaler = preprocess_data(train_df, is_train=True)

        # 创建数据集
        dataset = TensorDataset(train_data, train_labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        # 创建数据加载器
        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # 初始化模型
        print("\n初始化模型...")
        model = SurvivalClassifier(input_dim=6)  # 6个输入特征

        # 损失函数与优化器
        pos_weight = torch.tensor([train_df['Survived'].value_counts()[0] / train_df['Survived'].value_counts()[1]])
        criterion = nn.BCELoss()  # 使用BCELoss需确保输出经过Sigmoid
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        # 训练模型
        print("\n开始模型训练...")
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=100
        )

        # 测试集预测
        print("\n进行测试集预测...")
        test_data, _, _ = preprocess_data(test_df, imputer=imputer, scaler=scaler, is_train=False)

        # 加载最佳模型
        trained_model.load_state_dict(torch.load('best_model.pth'))
        trained_model.eval()

        with torch.no_grad():
            outputs = trained_model(test_data)
            preds = (outputs >= 0.5).int().squeeze()

        # 生成提交文件
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': preds.cpu().numpy()
        })
        submission.to_csv('submission.csv', index=False)
        print("\n预测结果已保存至 submission.csv")

    except FileNotFoundError as fe:
        print(f"文件未找到错误: {str(fe)}")
        print("请确保 train.csv 和 test.csv 位于当前目录")
    except Exception as e:
        print(f"程序运行异常: {str(e)}")