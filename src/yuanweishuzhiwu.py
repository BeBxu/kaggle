#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
iris = load_iris();
# print iris.data.shape
# print iris.DESCR
#这个数据集是按照类别依次排列，要保证随机采样
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
#KNN Classifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)
#Performance
print 'The accuracy of K-Nearest Neighbor Classifer is', knc.score(X_test, y_test)
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=iris.target_names)

