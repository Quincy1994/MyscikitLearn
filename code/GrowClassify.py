#coding=utf-8
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


## 读取数据
filename = "data/labelAttributes.txt"
data = load_svmlight_file(filename)
X, y = data[0], data[1]

## 标准化处理
scaler = preprocessing.StandardScaler(with_mean=False).fit(X)
X = scaler.transform(X)


## 划分数据集
rs = ShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4,random_state=0)
rs.get_n_splits(X)
X_trainset = None
y_trainset = None
X_testset = None
y_testset = None
for train_index, test_index in rs.split(X,y):
    X_trainset, X_testset = X[train_index], X[test_index]
    y_trainset, y_testset = y[train_index], y[test_index]


## 使用神经网络训练
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4,3), random_state=1)
clf.fit(X_trainset,y_trainset)


## 预测结果
result = clf.predict(X_testset)

## 计算准确率
count = 0
for i in range(0,result.__len__(),1):
    if result[i] == y_testset[i]:
        count += 1
print 1.0 * count / result.__len__()
