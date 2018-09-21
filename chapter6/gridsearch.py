"""
Grid-Searching Preprocessing and
Model Parameters
----------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
  PolynomialFeatures,
  StandardScaler,
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pipelines import POWERS_OF_TEN


boston = load_boston()
X_train, X_test, y_train, y_test = \
  train_test_split(boston['data'], boston['target'], random_state=0)


pipe = make_pipeline(
  StandardScaler(), PolynomialFeatures(), Ridge())
param_grid = {
  'polynomialfeatures__degree': [1, 2, 3],
  'ridge__alpha': POWERS_OF_TEN,
}
grid = GridSearchCV(
  pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
# plt.matshow(
#   grid.cv_results_['mean_test_score'].reshape(3, -1),
#   vmin=0,
#   cmap='viridis',
# )
# plt.xlabel('ridge__alpha')
# plt.ylabel('polynomialfeatures__degree')
# plt.xticks(
#   range(len(param_grid['ridge__alpha'])),
#   param_grid['ridge__alpha'],
# )
# plt.yticks(
#     range(len(param_grid['polynomialfeatures__degree'])),
#   param_grid['polynomialfeatures__degree'],
# )
# plt.colorbar()
# plt.show()
# Plot a heatmap comparing performance across
# different combinations of features in the different
# steps


# print 'Best parameters: {}'.format(grid.best_params_)
# {'ridge__alpha': 10, 'polynomialfeatures__degree': 2}
# print 'Test set score: {:.2f}'.format(grid.score(X_test, y_test))
# Scores 0.77
# Printing some attributes of the grid search on the pipeline


param_grid = {'ridge__alpha': POWERS_OF_TEN}
pipe = make_pipeline(StandardScaler(), Ridge())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
# print 'Score without poly features: {:.2f}'.format(grid.score(X_test, y_test))
# Without polynomial features the model does worse, only 0.63


pipe = Pipeline([
  ('preprocessing', StandardScaler()),
  ('classifier', SVC()),
])
param_grid = [
  {
    'classifier': [SVC()],
    'preprocessing': [StandardScaler(), None],
    'classifier__gamma': POWERS_OF_TEN,
    'classifier__C': POWERS_OF_TEN,
  },
  {
    'classifier': [RandomForestClassifier(n_estimators=100)],
    'preprocessing': [None],
    'classifier__max_features': [1, 2, 3],
  },
]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = \
  train_test_split(cancer['data'], cancer['target'], random_state=0)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
# print 'Best params:\n{}'.format(grid.best_params_)
# print 'Best cross-validation score: {:.2f}'.format(grid.best_score_)
# print 'Test-set score: {:.2f}'.format(grid.score(X_test, y_test))
# This grid search compares using a random forest classifier with
# a varied max_features parameter against a SVC with the data
# scaled using standard scaler.
