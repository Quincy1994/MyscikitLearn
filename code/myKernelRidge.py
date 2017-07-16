# coding=utf-8
from sklearn.kernel_ridge import KernelRidge


# 训练样本的特征空间
X = [[0,1],[0,2],[0,3],[1,0],[2,0],[3,0]]
# 训练样本的目标属性
y = [0.1,0.2,0.3,1.1,1.2,1.3]

# 构建SVM分类器, 将probability设为true,可以计算每个样本到各个类别的概率
clf = KernelRidge(alpha=1.0)
# SVM分类器训练样本
clf.fit(X,y)

# 测试样本
test = [[0,2.5],[2.5,0]]
# 用分类器预测测试样本, 计算每个样本到各个类别的概率
result = clf.predict(test)
print result