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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
  train_test_split,
  GridSearchCV,
  cross_val_score,
)
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA


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


def fit(self, X, y):
  """
  Implementation of a Pipeline's fit method
  All but the last step must have a fit_transform method
  Then the classifier fits the transformed data.

  Think of this as composing transformations before
  using that data to train a classifier

  """
  X_transformed = X
  for _, estimator in self.steps[:-1]:
    X_transformed = estimator.fit_transform(X_transformed, y)
  self.steps[-1][1].fit(X_transformed, y)


def predict(self, X):
  """
  Implementation of predict in a general
  pipeline interface

  """
  X_transformed = X
  for _, estimator in self.steps[:-1]:
    X_transformed = estimator.fit_transform(X_transformed)
  return self.steps[-1][1].predict(X_transformed)


# Standard syntax
pipe_long = Pipeline([
  ('scaler', MinMaxScaler()),
  ('svm', SVC(C=100)),
])
# Abbreviated syntax
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
# print 'Pipleline steps:\n{}'.format(pipe_short.steps)
# Has automatic naming


pipe = make_pipeline(
  StandardScaler(),
  PCA(n_components=2),
  StandardScaler(),
)
# print 'Pipeline steps:\n{}'.format(pipe.steps)
# Numbers are attached when the same type of estimator is used


pipe.fit(cancer['data'])
components = pipe.named_steps['pca'].components_
# print 'components.shape: {}'.format(components.shape)
# You can access the estimates at any step in the pipeline


pipe = make_pipeline(
  StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': POWERS_OF_TEN}
X_train, X_test, y_train, y_test = \
  train_test_split(cancer['data'], cancer['target'], random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
# print 'Best estimator:\n{}'.format(grid.best_estimator_)
# You can run grid search on piplelines and then get the pipeline
# with the best performance on the training and validation step


# print 'Logistic regression step:\n{}'.format(
#   grid.best_estimator_.named_steps['logisticregression'])
# You also have access to each step in the best performing pipeline


# print 'Logistic regression coefficients:\n{}'.format(
#   grid.best_estimator_.named_steps['logisticregression'].coef_)
# Also printing the coefficients for each feature in the dataset for the
# best performing logisticregression
