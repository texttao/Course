import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

#df=pd.DataFrame(datasets.load_iris()['data'],columns=datasets.load_iris()['feature_names'])

iris = datasets.load_iris()
X = iris.data[:, :4]  

# 绘制数据分布图
# plt.scatter(X[:, 0], X[:, 2], c="red", marker='o', label='see')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc=2)
# plt.show()

estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print(estimator.cluster_centers_)

# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 2], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 2], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 2], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc=2)
plt.show()  