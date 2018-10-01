"""
Parameter Selection with Preprocessing
--------------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


POWERS_OF_TEN = [10 ** i for i in range(-3, 3)]


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = \
  train_test_split(cancer['data'], cancer['target'], random_state=0)
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
svm = SVC()
svm.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
print 'Test score: {:.3f}'.format(svm.score(X_test_scaled, y_test))
# Example of using MinMaxScaler to preprocess cancer data and train
# an kernelized SVM


param_grid = {
  'C': POWERS_OF_TEN,
  'gamma': POWERS_OF_TEN,
}
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print 'Best cross-validation accuracy: {:.2f}'.format(grid.best_score_)
print 'Best parameters: {}'.format(grid.best_params_)
print 'Test set accuracy: {:.2f}'.format(grid.score(X_test_scaled, y_test))
# Example of using grid search to select parameters after scaling the data.
# The catch here is that the grid search splits the training set to test
# for the best parameters. But the scaling used all of the training set,
# which is fundamentally different from how the model gets new data.
# This training data has been included in the scaling


mglearn.plots.plot_improper_processing()
plt.show()
# This figure is an illustration of the incosistency described above
