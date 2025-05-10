import numpy as np
import pandas as pd


df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))
# 用 0 替换所有 NaN 元素。
print(df.fillna(0))

# 将“A”、“B”、“C”和“D”列中的所有 NaN 元素分别替换为 0、1、2 和 3。
values = {"A": 0, "B": 1, "C": 2, "D": 3}
print(df.fillna(value=values))