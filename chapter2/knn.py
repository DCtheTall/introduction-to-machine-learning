"""
k-Nearest Neighbors

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_breast_cancer


# mglearn.plots.plot_knn_classification(n_neighbors=1)
# plt.show()


# mglearn.plots.plot_knn_classification(n_neighbors=3)
# plt.show()


# Classifier for the forge dataset
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
# print 'Test set predictions: {}'.format(classifier.predict(X_test))
# print 'Test set accuracy: {:.2f}'.format(classifier.score(X_test, y_test))


# Plotting decision boundaries
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# for n_neighbors, ax in zip([1, 3, 9], axes):
#   classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#   mglearn.plots.plot_2d_separator(classifier, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#   mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
#   ax.set_title('{} neighbor(s)'.format(n_neighbors))
#   ax.set_xlabel('feature 0')
#   ax.set_ylabel('feature 1')
# axes[0].legend(loc=3)
# plt.show()


# Comparing model accuracy for different numbers of nearest neighbors
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
  cancer['data'],
  cancer['target'],
  stratify=cancer['target'],
  random_state=66,
)
train_accuracy = []
test_accuracy = []
neighbor_settings = range(1, 11)
for n_neighbors in neighbor_settings:
  classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
  classifier.fit(X_train, y_train)
  train_accuracy.append(classifier.score(X_train, y_train))
  test_accuracy.append(classifier.score(X_test, y_test))
# plt.plot(neighbor_settings, train_accuracy, label='training accuracy')
# plt.plot(neighbor_settings, test_accuracy, label='test acccuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('n_neighbors')
# plt.legend()
# plt.show()


# Plotting mglearn examples for kNN regressor
# mglearn.plots.plot_knn_regression(n_neighbors=1)
# mglearn.plots.plot_knn_regression(n_neighbors=3)
# plt.show()


# Implementing kNN regressor with scikit-learn
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X_train, y_train)
# print 'Test set predictions:\n{}'.format(regressor.predict(X_test))
# print 'Test set R^2: {}'.format(regressor.score(X_test, y_test))


# Analyzing accuracy of kNN regressor
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
  regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
  regressor.fit(X_train, y_train)
  ax.plot(line, regressor.predict(line))
  ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
  ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
  ax.set_title(
    '{} neighbors\n train score: {:.2f} test score: {:.2f}'.format(
      n_neighbors,
      regressor.score(X_train, y_train),
      regressor.score(X_test, y_test)
    )
  )
  ax.set_xlabel('Feature')
  ax.set_ylabel('Target')
axes[0].legend([
  'Model predictions',
  'Training data/target',
  'Test data/target',
], loc='best')
plt.show()
# Comparing regression when looking at varying numbers of
# nearest neighboars
