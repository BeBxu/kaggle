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
# KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_test)
#Performance
#ARI:数据本身带有正确的类别信息
from sklearn.metrics import adjusted_rand_score
print adjusted_rand_score(y_test, y_pred)
#Silhouette Coefficient:数据无所属类别信息，值域[-1,1]，越大越好



