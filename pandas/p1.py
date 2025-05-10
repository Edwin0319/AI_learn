import pandas as pd


df=pd.DataFrame(dict(x=range(5),y=range(2,7),z=range(4,9)),index=list('abcde'))

print(df)
print(df[df['x']>1])
# 刪除label為x的列，把修改完的表return出去
dt = df.drop(labels='x', axis=1, inplace=False) #inplace为True表示直接对原表修改。
print(dt)

# 刪除label為b的行，直接在原表修改
df.drop(labels='b', axis=0, inplace=True)
print(df)


