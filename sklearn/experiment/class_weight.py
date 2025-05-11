from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# ���ɶ�������ݣ�������5����𣬱�ǩ0~4��
X, y = ...  # �������

# ��ʽ1���Զ�����Ȩ�أ������Ƶ�ʷ��ȣ�
classes = [0, 1, 2, 3, 4]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# ��ʽ2���ֶ�ָ��Ȩ�أ��������2����Ҫ��
# class_weight_dict = {0: 0.1, 1: 0.1, 2: 0.5, 3: 0.2, 4: 0.1}

# ��ʼ��������������Ȩ��
clf = RandomForestClassifier(class_weight=class_weight_dict)
clf.fit(X, y)