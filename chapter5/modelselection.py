"""
Using Evaluation Metrics in Model Selection
-------------------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.svm import SVC
from sklearn.model_selection import (
  cross_val_score,
  train_test_split,
  GridSearchCV,
)
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import SCORERS


digits = load_digits()
# Default scoring is using model accuracy
print 'Default scoring: {}'.format(
  cross_val_score(SVC(), digits['data'], digits['target'] == 9))
# Setting scoring='accuracy' does not change result
explicit_accuracy = cross_val_score(
  SVC(), digits['data'], digits['target'] == 9, scoring='accuracy')
print 'Explicit accuracy scoring: {}'.format(explicit_accuracy)
roc_auc = cross_val_score(
  SVC(), digits['data'], digits['target'] == 9, scoring='roc_auc')
print 'AUC scoring: {}'.format(roc_auc)
# Example of how you can use the different metrics to select parameters
# for the model


X_train, X_test, y_train, y_test = \
  train_test_split(digits['data'], digits['target'] == 9, random_state=0)
param_grid = {'gamma': [.0001, .01, 1, 10]}
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print 'Grid Search with accuracy'
print 'Best parameters: {}'.format(grid.best_params_)
print 'Best cross-validation score (accuracy): {:.3f}'.format(grid.best_score_)
print 'Test set AUC: {:.3f}'.format(
  roc_auc_score(y_test, grid.decision_function(X_test)))
print 'Test set accuracy: {:.3f}'.format(grid.score(X_test, y_test))
# Grid Search with accuracy
# Best parameters: {'gamma': 0.0001}
# Best cross - validation score(accuracy): 0.970
# Test set AUC: 0.992
# Test set accuracy: 0.973


grid = GridSearchCV(
  SVC(), param_grid=param_grid, scoring='roc_auc')
grid.fit(X_train, y_train)
print 'Grid Search with AUC'
print 'Best parameters: {}'.format(grid.best_params_)
print 'Best cross-validation score (accuracy): {:.3f}'.format(grid.best_score_)
print 'Test set AUC: {:.3f}'.format(
    roc_auc_score(y_test, grid.decision_function(X_test)))
print 'Test set accuracy: {:.3f}'.format(grid.score(X_test, y_test))
# Grid Search with AUC
# Best parameters: {'gamma': 0.01}
# Best cross - validation score(accuracy): 0.997
# Test set AUC: 1.000
# Test set accuracy: 1.000
# Here we see using AUC on imbalanced data let to a better AUC score
# and even a better accuracy score


print 'Available scores:\n{}'.format(sorted(SCORERS.keys()))
# Different scoring metrics available
