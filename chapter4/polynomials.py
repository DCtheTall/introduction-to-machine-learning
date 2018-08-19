"""
Interactions and Polynomials
----------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)


# include_bias defaults=True adds a feature that is constantly 1
poly = PolynomialFeatures(degree=10, include_bias=False).fit(X)
X_poly = poly.transform(X)
# print 'X_poly.shape: {}'.format(X_poly.shape)
# Creates 10 features for each data point


# print 'Entries of X:\n{}'.format(X[:5])
# print 'Entries of X_poly:\n{}'.format(X_poly[:5])
# print 'Polynomial feature names:\n{}'.format(poly.get_feature_names())
# Each feature in the polynomial set is the original input
# feature raised to an integer power in [1, 10]


reg = LinearRegression().fit(X_poly, y)
line_poly = poly.transform(line)
# plt.plot(line, reg.predict(line_poly), label='Polynomial linear regression')
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel('Regression output')
# plt.xlabel('Input feature')
# plt.legend(loc='best')
# plt.show()
# Using the polynomial features with linear rgression yields the classical
# model of polynomial regression. This yields a better result than
# linear regression but does not perform well at the extremes of the dataset


for gamma in [1, 10]:
  svr = SVR(gamma=gamma).fit(X, y)
#   plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))
# plt.plot(X[:,0], y, 'o', c='k')
# plt.ylabel('Regression output')
# plt.xlabel('Input feature')
# plt.legend(loc='best')
# plt.show()
# SVM provides a better polynomial regression than a linear
# model with polynomial features


boston = load_boston()
X_train, X_test, y_train, y_test = \
  train_test_split(boston.data, boston.target, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
# print 'X_train.shape: {}'.format(X_train.shape)
# print 'X_train_poly.shape: {}'.format(X_train_poly.shape)
# The data originally had 13 features, which were expanded into
# 105 features
# print 'Polynomial feature names:\n{}'.format(poly.get_feature_names())
# The new features include individual features raised to the
# first and 2nd power and also combinations of different features
# multiplied together


ridge = Ridge().fit(X_train_scaled, y_train)
# print 'Score without interactions: {:.3f}'.format(ridge.score(X_test_scaled, y_test))
# Score w/o interactions is 0.621
ridge = Ridge().fit(X_train_poly, y_train)
# print 'Score with interactions: {:.3f}'.format(ridge.score(X_test_poly, y_test))
# Score w/ interactions is 0.763
# Here adding interactions improved the performance of the Ridge regressor


rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
print 'Score without interactions: {:.3f}'.format(rf.score(X_test_scaled, y_test))
# Score w/o interactions is 0.787
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
print 'Score with interactions: {:.3f}'.format(rf.score(X_test_poly, y_test))
# Score w/ interactions is 0.771
# For the more complex random forest regressor, adding interactions actually
# made the model perform worse. Random forest already outperformed the linear
# model without the need to add interactions

