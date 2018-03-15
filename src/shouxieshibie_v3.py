#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as plt
import pandas as pd

digits_train = pd.read_csv('../../Datasets/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('../../Datasets/optdigits/optdigits.tes', header=None)

X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
# print np.arange(64)
# print X_train
# print y_train
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]
# print X_test
# print y_test
#基于线性假设的支持向量机分类器
from sklearn.svm import LinearSVC
#用原始的64维度数据建模
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
#PCA压缩到20维度
from sklearn.decomposition import PCA
estimator = PCA(n_components=20)

pca_X_train = estimator.fit_transform(X_train)
pca_X_test = estimator.transform(X_test)

pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

#performance分析
from sklearn.metrics import classification_report
print 'The Accuracy of Linear SVC is', svc.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names=np.arange(10).astype(str))

print 'The Accuracy of PCA SVC is', pca_svc.score(pca_X_test, y_test)
print classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str))



