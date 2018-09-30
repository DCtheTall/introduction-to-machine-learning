"""
Grid Search
-----------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import (
  train_test_split,
  cross_val_score,
  GridSearchCV,
  ParameterGrid,
  StratifiedKFold,
)


POWERS_OF_TEN = [10 ** n for n in range(-3, 3)]


iris = load_iris()
X_train, X_test, y_train, y_test = \
  train_test_split(iris['data'], iris['target'], random_state=0)
print 'Size of training set: {} Size of the test set: {}'.format(
  X_train.shape[0], X_test.shape[0])
best_score = 0
for gamma in POWERS_OF_TEN:
  for C in POWERS_OF_TEN:
    svm = SVC(gamma=gamma, C=C).fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    if score > best_score:
      best_score = score
      best_parameters = {'C': C, 'gamma': gamma}
print 'Best score: {}'.format(best_score)
print 'Best parameters: {}'.format(best_parameters)
# Here we are finding the best tuning parameters for the support
# vector classifier on the iris dataset, finding the parameters
# which lead to the best performance on the test set


X_trainval, X_test, y_trainval, y_test = \
  train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = \
  train_test_split(X_trainval, y_trainval, random_state=1)
set_size = lambda x: x.shape[0]
print ('Size of training set: {}' \
  + ' Size of validation set: {} ' \
  + ' Size of test: {}').format(
    set_size(X_train), set_size(X_valid), set_size(X_test))
best_score = 0
for gamma in POWERS_OF_TEN:
  for C in POWERS_OF_TEN:
    svm = SVC(gamma=gamma, C=C).fit(X_train, y_train)
    score = svm.score(X_valid, y_valid)
    if score > best_score:
      best_score = score
      best_parameters = {'C': C, 'gamma': gamma}
svm = SVC(**best_parameters).fit(X_trainval, y_trainval)
best_test_score = svm.score(X_test, y_test)
print 'Best validation score: {:.3f}'.format(best_score)
print 'Best parameters: {}'.format(best_parameters)
print 'Test set accuracy with best parameters: {:.3f}'.format(best_test_score)
# Here the dataset is split into 3 sets:
# - The training set, same use as always
# - The validation set, used as a test set to find best params
# - The test set, used for measuring performance after optimizing params
#   and retraining using the union of the training and validation sets


best_score = 0
for gamma in POWERS_OF_TEN:
  for C in POWERS_OF_TEN:
    svm = SVC(gamma=gamma, C=C)
    scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
    score = np.mean(scores)
    if score > best_score:
      best_score = score
      best_parameters = {'C': C, 'gamma': gamma}
svm = SVC(**best_parameters).fit(X_trainval, y_trainval)
print 'Test set accuracy with best parameters: {:.3f}'.format(best_test_score)
# Here we can use cross validation to select the best
# set of parameters with a varied splitting of the test/validation set
# The drawback is this is very computationally complex, as it
# trains 5 models for each element in the parameter grid


mglearn.plots.plot_grid_search_overview()
plt.show()
# Flow diagram of grid search (see page 267)


param_grid = {
  'C': POWERS_OF_TEN,
  'gamma': POWERS_OF_TEN,
}
print 'Parameter grid:\n{}'.format(param_grid)
# One can construct a parameter grid using a dictionary like above


grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = \
  train_test_split(iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)
print 'Test set score: {:.2f}'.format(grid_search.score(X_test, y_test))
# sklearn has a class GridSearchCV which can abstracts away performing
# a grid search. Fitting the data with grid search trains a model per combination
# in the param grid. You can then use .score to test results with the model
# it finds with the best parameter combination


print 'Best parameters: {}'.format(grid_search.best_params_)
print 'Best validation score: {:.2f}'.format(grid_search.best_score_)
# It also stores the best validation score and the best parameters


print 'Best estimator:\n{}'.format(grid_search.best_estimator_)
# It also provides access to the model which performed the best on the validation set


results = pd.DataFrame(grid_search.cv_results_)
display(results.head())
# The grid search also contains all the data about the search in a dictionary
# which can be converted to a data frame


scores = np.array(results.mean_test_score).reshape(6, 6)
mglearn.tools.heatmap(
  scores,
  xlabel='gamma',
  xticklabels=param_grid['gamma'],
  ylabel='C',
  yticklabels=param_grid['C'],
  cmap='viridis',
)
plt.show()
# Plots a grid with the mean test score for each combination
# of parameters, as we see from the grid, the parameters greatly
# influence the accuracy of SVC


fig, axes = plt.subplots(1, 3, figsize=(13, 5))
param_grid_linear = {
  'C': np.linspace(1, 2, 6),
  'gamma': np.linspace(1, 2, 6),
}
param_grid_one_log = {
  'C': np.linspace(1, 2, 6),
  'gamma': np.logspace(-3, 2, 6),
}
param_grid_range = {
  'C': np.logspace(-3, 2, 6),
  'gamma': np.logspace(-7, -2, 6),
}
for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range], axes):
  grid_search = GridSearchCV(SVC(), param_grid, cv=5)
  grid_search.fit(X_train, y_train)
  scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)
  scores_image = mglearn.tools.heatmap(
    scores,
    xlabel='gamma',
    ylabel='C',
    xticklabels=param_grid['gamma'],
    yticklabels=param_grid['C'],
    cmap='viridis',
    ax=ax
  )
plt.colorbar(scores_image, ax=axes.tolist())
plt.show()
# Plotting a grid with the different mean test scores
# for ranges of the parameters that are not as helpful
# as the example above


param_grid = [
  {
    'kernel': ['linear'],
    'C': POWERS_OF_TEN,
  },
  {
    'kernel': ['rbf'],
    'C': POWERS_OF_TEN,
    'gamma': POWERS_OF_TEN,
  },
]
print 'List of grids:\n{}'.format(param_grid)
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print 'Best parameters: {}'.format(grid_search.best_params_)
print 'Best cross-validation score: {:.2f}'.format(grid_search.best_score_)
# Best parameters: {'C': 100, 'kernel': 'rbf', 'gamma': 0.01}
# Best cross-validation score: 0.97
results = pd.DataFrame(grid_search.cv_results_)
display(results.T)
# You can also do grid-searches over parameter sets that are
# not grids, just dictionaries with different settings
# In this example, we examine when the kernel of the SVC is
# linear or uses rbf, the former case does not use the gamma
# parameter so it was omitted


param_grid = {
  'C': POWERS_OF_TEN,
  'gamma': POWERS_OF_TEN,
}
scores = cross_val_score(
  GridSearchCV(SVC(), param_grid, cv=5),
  iris.data,
  iris.target,
  cv=5,
)
print 'Cross-validation scores: {}'.format(scores)
print 'Mean cross-validation score: {}'.format(scores.mean())
# This example uses cross_val_score and then a nested GridSearch
# to perform a grid search on different splittings of the original data.
# This is a good test of how the model generalizes to new data
# since before our test results depended on how we split the data


def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
  """
  A simplified implementation of the cross validation score
  method call with a nested grid search

  """
  outer_scores = []
  for training_samples, test_samples in outer_cv.split(X, y):
    best_params = {}
    best_score = -np.inf
    for parameters in parameter_grid:
      cv_scores = []
      for inner_train, innter_test in inner_cv.split(
        X[training_samples],
        y[training_samples],
      ):
        clf = Classifier(**parameters)
        clf.fit(X[inner_train], y[inner_train])
        score = clf.score(X[innter_test], y[innter_test])
        cv_scores.append(score)
      mean_score = np.mean(cv_scores)
      if mean_score > best_score:
        best_score = mean_score
        best_params = parameters
    clf = Classifier(**best_params)
    clf.fit(X[training_samples], y[training_samples])
    outer_scores.append(clf.score(X[test_samples], y[test_samples]))
  return np.array(outer_scores)


scores = nested_cv(
  iris.data,
  iris.target,
  StratifiedKFold(5),
  StratifiedKFold(5),
  SVC,
  ParameterGrid(param_grid),
)
print 'Cross validation scores:\n{}'.format(scores)
# Testing the implementation of nested cross validation


param_grid = {
  'C': POWERS_OF_TEN,
  'gamma': POWERS_OF_TEN,
}
scores = cross_val_score(
  GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1),
  iris.data,
  iris.target,
  cv=5,
)
print 'Cross-validation scores: {}'.format(scores)
print 'Mean cross-validation score: {}'.format(scores.mean())
# You can speed up cross validation by using multiple cores
# Set the n_jobs parameter in GridSearchCV to parallelize the process
# over n CPU cores (n > 0) or all available cores (n = -1)
# This cannot be done if the model also parallelizes
# over multiple CPU cores as well (i.e. RandomForest)
