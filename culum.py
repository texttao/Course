import numpy as np
from sklearn import datasets
from sklearn import tree
import pydotplus

X, y = datasets.load_iris(return_X_y=True) #X与y
target_names=datasets.load_iris().target_names #y的值列表:0:setosa,1:versicolor,2:virginica
feature_names=datasets.load_iris().feature_names #特征X的名称列表

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

X_test = np.array([[6, 1, 3, 1]])
label = clf.predict(X_test)
print('class predict:', target_names[label[0]])

data_feature_name = feature_names
data_target_name = np.unique(target_names)

dot_tree = tree.export_graphviz(clf,out_file=None,feature_names=data_feature_name,class_names=data_target_name,
                                filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_tree)
graph.write_pdf("treeone.pdf")     # 经分析依据petal length (cm) ≤ 2.45， 类别最可能为setosa

