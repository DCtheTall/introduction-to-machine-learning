"""
Linear regression model
-----------------------

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


mglearn.plots.plot_linear_regression_wave()
plt.show()


X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print 'linreg.coef_: {}' .format(lr.coef_)
print 'linreg.intercept_: {}'.format(lr.intercept_)
print 'Training set score: {:.2f}'.format(lr.score(X_train, y_train))
print 'Test set score: {:.2f}'.format(lr.score(X_test, y_test))
# R^2 = 0.66 - Underfit b/c only 1 feature


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print 'Training set score: {:.2f}'.format(lr.score(X_train, y_train)) # 0.95
print 'Test set set score: {:.2f}'.format(lr.score(X_test, y_test)) # 0.61
# Training score >> test score which is a sign of an overfit
