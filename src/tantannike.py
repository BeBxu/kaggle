#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
titanic = pd.read_csv('../../Datasets/Titanic/titanic.txt')
# print titanic.head()
#查看数据统计特性
# titanic.info()
#特征选择
X=titanic[['pclass','age','sex']]
y=titanic['survived']
# X.info()
#补完age列
X['age'].fillna(X['age'].mean(), inplace=True)
# X.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# sex与pclass从object转化为数值类型，0/1
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
# ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
# print vec.feature_names_
X_test = vec.transform(X_test.to_dict(orient='record'))
#单一决策树
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred = dtc.predict(X_test)
#随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred = rfc.predict(X_test)
#梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred = gbc.predict(X_test)

#Performance
from sklearn.metrics import classification_report
print 'The accuracy of DecisionTreeClassifier is', dtc.score(X_test, y_test)
print classification_report(y_test, dtc_y_pred)

print 'The accuracy of random forest classifer is', rfc.score(X_test,y_test)
print classification_report(y_test, rfc_y_pred)

print 'The accuracy of gradient boosting tree is', gbc.score(X_test,y_test)
print classification_report(y_test, gbc_y_pred)






