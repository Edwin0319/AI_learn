import torch
from torch import nn


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # ����ѵ�����ݣ�������ʽ������ֱ�Ӵ洢������
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # �����������������ѵ��������ŷ�Ͼ��루�㲥���٣�
        distances = torch.cdist(X_test, self.X_train, p=2)  # (n_test, n_train)

        # ��ȡǰk������ڵ������ͱ�ǩ
        _, topk_indices = torch.topk(distances, self.k, largest=False)  # (n_test, k)
        topk_labels = self.y_train[topk_indices]  # (n_test, k)

        # ͳ�ƶ���ͶƱ���
        preds, _ = torch.mode(topk_labels, dim=1)  # (n_test,)
        return preds


# ����ʾ��
if __name__ == "__main__":
    # ����ʾ�����ݣ�ȷ���ɸ��֣�
    torch.manual_seed(42)
    X = torch.randn(100, 2)  # 100��������2������
    y = torch.randint(0, 2, (100,))  # �������ǩ

    # ����ѵ�����Ͳ��Լ�
    train_idx = torch.randperm(100)[:80]
    test_idx = torch.randperm(100)[80:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # ѵ����Ԥ��
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    # ����׼ȷ��
    accuracy = (preds == y_test).float().mean()
    print(f"[PyTorch KNN] ׼ȷ��: {accuracy.item():.4f}")