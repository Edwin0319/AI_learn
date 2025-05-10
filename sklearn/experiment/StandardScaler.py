from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import numpy as np

data = np.array([[1, 2], [3, 4], [5, 6]])

# 標準化
# 数据处理成均值为0，方差为1的数据。优点是受异常值影响较小。公式为：(X-μ)/σ
# StandardScaler将数据进行标准化处理，使得数据的均值为0，方差为1。具体做法是对每个特征的数值进行减去该特征的均值，然后除以该特征的标准差。
# 創造StandardScaler對像
scaler1 = StandardScaler()
# 對數據進行標準化處理
scaled_data1 = scaler1.fit_transform(data)
print(scaled_data1)


# 歸一化
# 数据缩放到[0,1]范围内，缺点是受极值影响。公式为(X-min)/(max-min)。
# MinMaxScaler将数据进行缩放处理，使得数据的范围在指定的最小值和最大值之间。具体做法是对每个特征的数值进行线性变换，使得该特征的最小值变为0，最大值变为1。
# MinMaxScaler
scaler2 = MinMaxScaler()
# 對數據進行縮放處理
scaled_data2 = scaler2.fit_transform(data)
print(scaled_data2)

# sklearn 不接受張量輸入
# 保持數據為 numpy array 直至完成所有 sklearn 預處理（包括標準化）
