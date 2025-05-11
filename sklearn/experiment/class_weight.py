from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# 生成多分类数据（假设有5个类别，标签0~4）
X, y = ...  # 你的数据

# 方式1：自动计算权重（按类别频率反比）
classes = [0, 1, 2, 3, 4]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# 方式2：手动指定权重（例如类别2最重要）
# class_weight_dict = {0: 0.1, 1: 0.1, 2: 0.5, 3: 0.2, 4: 0.1}

# 初始化分类器并传入权重
clf = RandomForestClassifier(class_weight=class_weight_dict)
clf.fit(X, y)