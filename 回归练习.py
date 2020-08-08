
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data[:, :14]
Y = boston.target[:]
feature_names = boston.feature_names[:]
x_axis_name = ['RM','DIS','PTRATIO','LSTAT']
# 绘制数据分布图
index_list = []
for name in x_axis_name:
    index = feature_names.tolist().index(name)
    index_list.append(index)
    plt.scatter(X[:, index], Y, c="red", marker='o', label='see')
    plt.xlabel('RM')
    plt.ylabel('PRICE')
    plt.legend(loc=2)
    plt.show()
x_train = X[:, index_list]
model = LinearRegression()
model.fit(x_train, Y)
linear_string = ''

for i, w in enumerate(model.coef_):
    linear_string += '{}*{}'.format(str(w), x_axis_name[i])
print('当前线性方程为 y = ' + str(model.intercept_) + '+' + linear_string)

x_test = [[8, 2, 12, 22]]
y_pred = model.predict(x_test)[0]
print('该小区房价约为 ', y_pred)
