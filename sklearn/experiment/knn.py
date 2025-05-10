import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 测试示例
if __name__ == "__main__":
    # 生成与PyTorch相同的数据（确保结果可比）
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100个样本，2个特征
    y = np.random.randint(0, 2, 100)  # 二分类标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 训练和预测
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, preds)
    print(f"[Scikit-learn KNN] 准确率: {accuracy:.4f}")