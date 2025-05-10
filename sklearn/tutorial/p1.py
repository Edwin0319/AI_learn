import sklearn
from sklearn import linear_model, datasets
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
# 缐性回歸
model = linear_model.LinearRegression()

data_X = np.array([[2.25, 0.85, 0.2, 0.45]])
data_Y = np.array([[0.35, 0.4, 0.61, 0.5]])
data_X = data_X.reshape(-1, 1)
data_Y = data_Y.reshape(-1, 1)
print(data_X, data_Y)
# train
model.fit(data_X, data_Y)
# 自变量参数
print(model.coef_)

test = np.array([[0, 4.8]])
test = test.reshape(-1, 1)
print(test)
prediction = model.predict(test)
print(prediction)

Y = np.array([[0.6], [0]])
mean_squared_error(prediction, Y)
print()

print(model.predict(test))









import matplotlib.pyplot as plt
plt.scatter(data_X, data_Y, color='blue', label='Training Data')

# 生成回归线
x_range = np.linspace(min(data_X.min(), test.min())-1,
                      max(data_X.max(), test.max())+1, 100).reshape(-1, 1)
y_range = model.predict(x_range)
plt.plot(x_range, y_range, color='red', linewidth=2, label='Regression Line')

# 绘制预测点
test_points = test.reshape(-1, 1)
predictions = model.predict(test_points)
plt.scatter(test_points, predictions, color='green', s=100, marker='X',
            label='Predictions', edgecolors='black')

# 标注预测值
for i, (x, y) in enumerate(zip(test_points, predictions)):
    plt.text(x, y, f'({x[0]:.1f}, {y[0]:.2f})', fontsize=9, ha='right')

# 设置图形参数
plt.xlabel('X Features')
plt.ylabel('Y Target')
plt.title('Linear Regression Prediction Visualization')
plt.grid(True, linestyle='--', alpha=0.7 ) #网格
plt.legend() # 右上角的label
plt.show()