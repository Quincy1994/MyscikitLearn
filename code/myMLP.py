# coding=utf-8
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import os

# 模型保存的文件夹路径
os.chdir("/home/quincy/model_save")

# 训练模型
X = [[0., 0.], [1., 1.],[2.,2.]]
y = [0,1,2]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,3), random_state=1)
clf.fit(X,y)

# 将训练好的模型保存到train_model.m中
joblib.dump(clf, "train_model.m")

# 模型的加载
clf = joblib.load("train_model.m")

test = [[2., 2.],[1., 2.]]
result = clf.predict_proba(test)
print result