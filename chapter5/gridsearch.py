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
from sklearn.model_selection import train_test_split, cross_val_score


iris = load_iris()
X_train, X_test, y_train, y_test = \
  train_test_split(iris.data, iris.target, random_state=0)
# print 'Size of training set: {} Size of the test set: {}'.format(
#   X_train.shape[0], X_test.shape[0])
best_score = 0
mags = [10 ** n for n in range(-3, 3)]
for gamma in mags:
  for C in mags:
    svm = SVC(gamma=gamma, C=C).fit(X_train, y_train)
    score = svm.score(X_test, y_test)
    if score > best_score:
      best_score = score
      best_parameters = {'C': C, 'gamma': gamma}
# print 'Best score: {}'.format(best_score)
# print 'Best parameters: {}'.format(best_parameters)
# Here we are finding the best tuning parameters for the support
# vector classifier on the iris dataset, finding the parameters
# which lead to the best performance on the test set


X_trainval, X_test, y_trainval, y_test = \
  train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = \
  train_test_split(X_trainval, y_trainval, random_state=1)
set_size = lambda x: x.shape[0]
# print ('Size of training set: {}' \
#   + ' Size of validation set: {} ' \
#   + ' Size of test: {}').format(
#     set_size(X_train), set_size(X_valid), set_size(X_test))
best_score = 0
for gamma in mags:
  for C in mags:
    svm = SVC(gamma=gamma, C=C).fit(X_train, y_train)
    score = svm.score(X_valid, y_valid)
    if score > best_score:
      best_score = score
      best_parameters = {'C': C, 'gamma': gamma}
svm = SVC(**best_parameters).fit(X_trainval, y_trainval)
best_test_score = svm.score(X_test, y_test)
# print 'Best validation score: {:.3f}'.format(best_score)
# print 'Best parameters: {}'.format(best_parameters)
# print 'Test set accuracy with best parameters: {:.3f}'.format(best_test_score)
# Here the dataset is split into 3 sets:
# - The training set, same use as always
# - The validation set, used as a test set to find best params
# - The test set, used for measuring performance after optimizing params
#   and retraining using the union of the training and validation sets


best_score = 0
for gamma in mags:
  for C in mags:
    svm = SVC(gamma=gamma, C=C)
    scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
    score = np.mean(scores)
    if score > best_score:
      best_score = score
      best_parameters = {'C': C, 'gamma': gamma}
svm = SVC(**best_parameters).fit(X_trainval, y_trainval)
# print 'Test set accuracy with best parameters: {:.3f}'.format(best_test_score)
# Here we can use cross validation to select the best
# set of parameters with a varied splitting of the test/validation set
# The drawback is this is very computationally complex, as it
# trains 5 models for each element in the parameter grid


# mglearn.plots.plot_cross_val_selection()
# plt.show()
# TODO debug this?
