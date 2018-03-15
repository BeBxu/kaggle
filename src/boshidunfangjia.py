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
#1d array was expected, for example using ravel()
y_train = y_train.ravel()
y_test = y_test.ravel()


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
#Performance
print 'The value of default measurement of LinearRegression is', lr.score(X_test,y_test)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print 'The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict)
print 'The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))
print 'The mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))

print 'The value of default measurement of SGDRegressor is', sgdr.score(X_test,y_test)
print 'The value of R-squared of SGDRegressor is', r2_score(y_test,sgdr_y_predict)
print 'The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))
print 'The mean absolute error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))

#SVM回归模型
from sklearn.svm import SVR
#线性核函数
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)
#多项式核函数
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)
#径向基核函数
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

#Performance
print 'The value of default measurement of linear SVR is', linear_svr.score(X_test,y_test)
print 'The value of R-squared of linear SVR is', r2_score(y_test,linear_svr_y_predict)
print 'The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))
print 'The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict))

print 'The value of default measurement of poly SVR is', poly_svr.score(X_test,y_test)
print 'The value of R-squared of poly SVR is', r2_score(y_test,poly_svr_y_predict)
print 'The mean squared error of poly SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))
print 'The mean absolute error of poly SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict))

print 'The value of default measurement of RBF SVR is', rbf_svr.score(X_test,y_test)
print 'The value of R-squared of RBF SVR is', r2_score(y_test,rbf_svr_y_predict)
print 'The mean squared error of RBF SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))
print 'The mean absolute error of RBF SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict))

#KNN回归
from sklearn.neighbors import KNeighborsRegressor
#平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict = uni_knr.predict(X_test)
#加权回归
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict = dis_knr.predict(X_test)
#Performance
print 'The value of default measurement of uniform-weighted KNeighborRegression is', uni_knr.score(X_test,y_test)
print 'The value of R-squared of uniform-weighted KNeighborRegression is', r2_score(y_test,uni_knr_y_predict)
print 'The mean squared error of uniform-weighted KNeighborRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))
print 'The mean absolute error of uniform-weighted KNeighborRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict))

print 'The value of default measurement of distance-weighted KNeighborRegression is', dis_knr.score(X_test,y_test)
print 'The value of R-squared of distance-weighted KNeighborRegression is', r2_score(y_test,dis_knr_y_predict)
print 'The mean squared error of distance-weighted KNeighborRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))
print 'The mean absolute error of distance-weighted KNeighborRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict))

#单一的回归树
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict = dtr.predict(X_test)
print 'The value of default measurement of DecisionTreeRegressor is', dtr.score(X_test,y_test)
print 'The value of R-squared of DecisionTreeRegressor is', r2_score(y_test,dtr_y_predict)
print 'The mean squared error of DecisionTreeRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict))
print 'The mean absolute error of DecisionTreeRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict))

#集成回归模型
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
#普通随机森林
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict = rfr.predict(X_test)
#极端随机森林
etr = ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict = etr.predict(X_test)
#梯度提升
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)
#Performance
print 'The value of default measurement of RandomForestRegressor is', rfr.score(X_test,y_test)
print 'The value of R-squared of RandomForestRegressor is', r2_score(y_test,rfr_y_predict)
print 'The mean squared error of RandomForestRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))
print 'The mean absolute error of RandomForestRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict))

print 'The value of default measurement of ExtraTreesRegressor is', etr.score(X_test,y_test)
print 'The value of R-squared of ExtraTreesRegressor is', r2_score(y_test,etr_y_predict)
print 'The mean squared error of ExtraTreesRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict))
print 'The mean absolute error of ExtraTreesRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict))

print 'The value of default measurement of GradientBoostingRegressor is', gbr.score(X_test,y_test)
print 'The value of R-squared of GradientBoostingRegressor is', r2_score(y_test,gbr_y_predict)
print 'The mean squared error of GradientBoostingRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict))
print 'The mean absolute error of GradientBoostingRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict))








