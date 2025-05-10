# missing_values：int, float, str, (默认)np.nan或是None, 即缺失值是什么。
# strategy：空值填充的策略，共四种选择（默认）mean、median、most_frequent、constant。mean表示该列的缺失值由该列的均值填充。median为中位数，most_frequent为众数。constant表示将空值填充为自定义的值，但这个自定义的值要通过fill_value来定义。
# fill_value：str或数值，默认为Zone。当strategy == "constant"时，fill_value被用来替换所有出现的缺失值（missing_values）。fill_value为Zone，当处理的是数值数据时，缺失值（missing_values）会替换为0，对于字符串或对象数据类型则替换为"missing_value" 这一字符串。
# verbose：int，（默认）0，控制imputer的冗长。
# copy：boolean，（默认）True，表示对数据的副本进行处理，False对数据原地修改。
# add_indicator：boolean，（默认）False，True则会在数据后面加入n列由0和1构成的同样大小的数据，0表示所在位置非缺失值，1表示所在位置为缺失值。
#


from sklearn.impute import SimpleImputer
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
X1 = np.array([[1, 2, np.nan],
               [4, np.nan, 6],
               [np.nan, 8, 9]])
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# 填补值为X矩阵各列的均值。
imp.fit(X)
print(imp.transform(X1))



X1 = np.array([[1, 2, np.nan],
               [4, np.nan, 6],
               [np.nan, 8, 9]])
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# 以列的平均值埴空, 每列剩餘的數的平均值
print(imp.fit_transform(X1))