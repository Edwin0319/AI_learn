import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ����ʾ��
if __name__ == "__main__":
    # ������PyTorch��ͬ�����ݣ�ȷ������ɱȣ�
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100��������2������
    y = np.random.randint(0, 2, 100)  # �������ǩ

    # ����ѵ�����Ͳ��Լ�
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ѵ����Ԥ��
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    # ����׼ȷ��
    accuracy = accuracy_score(y_test, preds)
    print(f"[Scikit-learn KNN] ׼ȷ��: {accuracy:.4f}")