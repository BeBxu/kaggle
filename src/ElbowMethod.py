#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#2个len：10的array：x1-x10 y1-y10
cluster1 = np.random.uniform(0.5,1.5,(2,10))
cluster2 = np.random.uniform(5.5,6.5,(2,10))
cluster3 = np.random.uniform(3.0,4.0,(2,10))
# print cluster1
#hstack分别合并三个len:10的array，x1-x30 .T是转置
X = np.hstack((cluster1,cluster2,cluster3)).T
# Y = np.hstack((cluster1,cluster2,cluster3))
# print X
# print X[:,0]
# print Y
# print Y[0]

plt.scatter(X[:,0], X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

K = range(1,10)
meandistortions=[]

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    dis = cdist(X,kmeans.cluster_centers_,'euclidean')
#     print 'dis:',dis
# axis=1:矩阵每一行向量
    min = np.min(dis,axis=1)
#     print 'min:',min
    avg = sum(min)/X.shape[0]
    meandistortions.append(avg)

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()
