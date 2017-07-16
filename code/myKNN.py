#coding=utf-8

from sklearn import neighbors
from sklearn.datasets import load_iris

n_neighbors = 5  # 最近邻的个数

# 读取数据集
iris = load_iris()
features = iris.data[:, :2]  # 特征矩阵
target = iris.target  # 目标属性

# 构建分类器
clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(features, target)

# 预测分类结果
pred = clf.predict(features)
print pred
