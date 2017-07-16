# coding=utf-8

from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import StratifiedKFold

# 原始输入空间
X = np.array([[1.,-1.,2.],
              [2.,0.,0.],
              [0.,1.,-1.]])

# 数据中心化
scaler = preprocessing.StandardScaler().fit(X)

# 构建特征空间
features = scaler.transform(X)
print features