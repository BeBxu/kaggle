#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from numpy import poly
boston = load_boston()
# print boston.DESCR
from sklearn.model_selection import train_test_split
import numpy as np
X = boston.data
y = boston.target
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,random_state=33)
#分析回归目标值差异
print 'The max target value is', np.max(y)
print 'The min target value is', np.min(y)
print 'The average target value is', np.mean(y)
#预测目标房价之间差异较大，所以做标准化，可用inverse_transform还原结果
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
#Passing 1d arrays as data is deprecated
# X.reshape(-1, 1) if your data has a single feature
# X.reshape(1, -1) if it contains a single sample.
y_train = ss_y.fit_transform(y_train.reshape(-1,1))
X_test = ss_X.transform(X_test)
y_test = ss_y.transform(y_test.reshape(-1,1))

print y_train.ravel()

#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)
#SGDRegressor
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
sgdr_y_predict = sgdr.predict(X_test)



