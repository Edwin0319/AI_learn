import pandas as pd
df=pd.DataFrame(dict(x=range(5),y=range(2,7),z=range(4,9)),index=list('abcde'))

# 刪掉原表label為x的列返回出去
poped_data = df.pop('x')
print(poped_data)

# 在index為2的列插入新值，名字為xed
df.insert(loc=2, column='xed', value=poped_data)
print(df)