# coding=utf-8
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
import numpy as np

X = [[0., 0.], [1., 1.],[2.,2.]]
y = [0,1,2]

X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(float)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,3), random_state=1)
clf.fit(X,y)

test = [[0.1, 0.1],[1.5,1.5],[2.5,2.5]]
test = PolynomialFeatures(interaction_only=True).fit_transform(test).astype(float)
result = clf.predict_proba(test)
print result
