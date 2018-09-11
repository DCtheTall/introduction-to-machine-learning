"""
Buidling Pipelines
and Using Pipelines in Grid Search
----------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
  train_test_split,
  GridSearchCV,
  cross_val_score,
)
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import Ridge


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = \
  train_test_split(cancer['data'], cancer['target'], random_state=0)
pipe = Pipeline([
  ('scaler', MinMaxScaler()),
  ('svm', SVC()),
])
pipe.fit(X_train, y_train)
# print 'Test score: {:.3f}'.format(pipe.score(X_test, y_test))
# This result is identical to applying MinMaxScaler to the
# training and test set and then scoring it using the kernelized SVM


POWERS_OF_TEN = [10 ** i for i in range(-3, 3)]
param_grid = {
  'svm__C': POWERS_OF_TEN,
  'svm__gamma': POWERS_OF_TEN,
}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
# print 'Best cross-validation: {:.3f}'.format(grid.best_score_)
# print 'Test set score: {:.3f}'.format(grid.score(X_test, y_test))
# print 'Best parameters: {}'.format(grid.best_params_)
# Here, the scaling is done properly during the grid search, instead
# of the whole training set being used, it uses the part of the
# training set it uses for training the different models for cross validation


# mglearn.plots.plot_proper_processing()
# plt.show()
# Illustration of proper preprocessing


rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))
# To illustrate information leakage, we start with 100 samples
# randomly chosen with 10,000 fatures. Because the data is just
# noise we should not be able to learn from it


select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
# print 'X_selected.shape: {}'.format(X_selected.shape)
# first select the most 500 relevant features
# print 'Cross validation accuracy (cv only on Ridge): {:.3f}'.format(
#   np.mean(cross_val_score(Ridge(), X_selected, y, cv=5)))
# The R^2 score is 0.91, this cannot be right since the data is just random.
# This is because we did preprocessing on the data outside the cross
# validation.


pipe = Pipeline([
  ('select', SelectPercentile(score_func=f_regression, percentile=5)),
  ('ridge', Ridge()),
])
# print 'Cross validation score accuracy (pipeline): {:.3f}'.format(
#   np.mean(cross_val_score(pipe, X, y, cv=5)))
# Here the R^2 score is -0.25, indicating a bad model, which is correct
