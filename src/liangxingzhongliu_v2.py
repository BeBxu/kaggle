#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from dask.array.tests.test_array_core import test_size
#数据预处理
column_names=['Sample code number','Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv('../../Datasets/Breast-Cancer/breast-cancer-wisconsin.data', names=column_names)
#将 ？ 替换为标准缺失值
data = data.replace(to_replace='?', value=np.nan)
#丢弃带有缺失值的数据，只要一个维度有缺失
data = data.dropna(how='any')
# print data
print data.shape
#准备训练和测试数据
# This module will be removed in 0.20.
# from sklearn.cross_validation import train_test_split
# 替代方法
from sklearn.model_selection import train_test_split
#random_state=33伪随机数产生器的种子，也就是“the starting point for a sequence of pseudorandom number”
#对于某一个伪随机数发生器，只要该种子（seed）相同，产生的随机数序列就是相同的.
# 如果你设置为 None，则会随机选择一个种子
#25%为test
# data[column_names[1:10]]：所要划分的样本特征集
# data[column_names[10]]：所要划分的样本结果
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
#value_counts还是一个顶级的pandas方法。可用于任何是数组或者序列
print y_train.value_counts()
print y_test.value_counts()
#使用LR和随机梯度参数估计
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
#Normalization:保证每个维度的特征数据方差为1均值为0，使得预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
# http://blog.csdn.net/quiet_girl/article/details/72517053
# https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn
# fit_transform()先拟合数据，再标准化  
X_train = ss.fit_transform(X_train)
# transform()数据标准化  只需要标准化数据而不需要再次拟合数据
X_test = ss.transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train, y_train,)
sgdc_y_predict = sgdc.predict(X_test)

#performance分析
from sklearn.metrics import classification_report
#自带评分函数
print 'Accuracy of LR Classifier:', lr.score(X_test, y_test)
print classification_report(y_test,lr_y_predict, target_names=['Benign','Malignant'])

print 'Accuracy of SGD Classifier:',sgdc.score(X_test,y_test)
print classification_report(y_test, sgdc_y_predict, target_names=['Benign','Malignant'])









