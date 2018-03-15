#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#read_csv读取的数据类型为Dataframe，obj.dtypes可以查看每列的数据类型
df_train = pd.read_csv('../../Datasets/Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('../../Datasets/Breast-Cancer/breast-cancer-test.csv')
# print df_test
# print type(df_test)
# print df_test.dtypes
#dataframe 行列选择，切片操作
# 1）loc，基于列label，可选取特定行（根据行index）； .loc仅支持列名操作,loc可以不加列名，则是行选择
# 2）iloc，基于行/列的position； iloc可以不加第几列，则是行选择
# 3）at，根据指定行index及列label，快速定位DataFrame的元素； 
# 4）iat，与at类似，不同的是根据position来定位的； 
# 5）ix，为loc与iloc的混合体，既支持label也支持position；
# df[]只能进行行选择，或列选择，不能同时进行列选择，列选择只能是列名。
# print df_test.loc[df_test['Type']==0, ['Clump Thickness','Cell Size']]
# print df_test.loc[0:3][['Clump Thickness','Cell Size']]
df_test_negative = df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive = df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]

plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
##########
intercept = np.random.random([1])
print intercept
#2维随机数
coef = np.random.random([2])
print coef
#根据start与stop指定的范围以及step设定的步长，生成一个序列。
lx = np.arange(0,12)
print lx
# lx * coef[0] + ly*coef[1] + intercept = 0
ly = (-intercept-lx*coef[0])/coef[1]
print ly

plt.plot(lx,ly, c='yellow')

plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
#####################
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print 'Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly, c='green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()
#############
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print 'Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])

intercept = lr.intercept_
coef = lr.coef_[0, :]
ly = (-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly, c='blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()


