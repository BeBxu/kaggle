#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
digits = load_digits()
# 8x8=64的像素矩阵表示
print digits.data.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.25, random_state=33)
print y_train.shape
print y_test.shape

#支持向量机
from sklearn.preprocessing import StandardScaler
#基于线性假设的支持向量机分类器
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
y_predict = lsvc.predict(X_test)

#performance分析
print 'The Accuracy of Linear SVC is', lsvc.score(X_test, y_test)

from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))

