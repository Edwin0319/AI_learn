import torch
from torch import nn


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # 保存训练数据（无需显式拷贝，直接存储张量）
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # 计算测试样本与所有训练样本的欧氏距离（广播加速）
        distances = torch.cdist(X_test, self.X_train, p=2)  # (n_test, n_train)

        # 获取前k个最近邻的索引和标签
        _, topk_indices = torch.topk(distances, self.k, largest=False)  # (n_test, k)
        topk_labels = self.y_train[topk_indices]  # (n_test, k)

        # 统计多数投票结果
        preds, _ = torch.mode(topk_labels, dim=1)  # (n_test,)
        return preds


# 测试示例
if __name__ == "__main__":
    # 生成示例数据（确保可复现）
    torch.manual_seed(42)
    X = torch.randn(100, 2)  # 100个样本，2个特征
    y = torch.randint(0, 2, (100,))  # 二分类标签

    # 划分训练集和测试集
    train_idx = torch.randperm(100)[:80]
    test_idx = torch.randperm(100)[80:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 训练和预测
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    # 计算准确率
    accuracy = (preds == y_test).float().mean()
    print(f"[PyTorch KNN] 准确率: {accuracy.item():.4f}")