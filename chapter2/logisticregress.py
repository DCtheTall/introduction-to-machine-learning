"""
Logistic Regression
and Linear SVC
--------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split


# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#   clf = model.fit(X, y)
#   mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
#   mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#   ax.set_title('{}'.format(clf.__class__.__name__))
#   ax.set_xlabel('Feature 0')
#   ax.set_ylabel('Feature 1')
# axes[0].legend()
# plt.show()
# Show decision boundaries for LogisticRegression and LinearSVC on forge data


# mglearn.plots.plot_linear_svc_regularization()
# plt.show()
# Show decision boundaries for LinearSVC with varied levels of
# the constant, C, which is inversely correlated to the amount
# of regularization


def print_model_score(model):
  print 'Training set score: {:.3f}'.format(model.score(X_train, y_train))
  print 'Test set score: {:.3f}'.format(model.score(X_test, y_test))


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
  cancer['data'],
  cancer['target'],
  stratify=cancer['target'],
  random_state=42,
)


logreg = LogisticRegression().fit(X_train, y_train)
# print_model_score(logreg)
# Training set score: .953
# Test set score: .958
# Test and training score are about the same: possibly an underfit


logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# print_model_score(logreg100)
# Training set score: .972
# Test set score: .965
# Increasing C, which decreased regularization, improved performance
# so our prediction was correct


logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# print_model_score(logreg001)
# Training set score: 0.934
# Test set score: 0.930
# This performed worse than the default (C=1), worse underfitting


# plt.plot(logreg.coef_.T, 'o', label='C=1')
# plt.plot(logreg100.coef_.T, '^', label='C=100')
# plt.plot(logreg001.coef_.T, 'v', label='C=0.01')
# plt.xticks(range(cancer['data'].shape[1]), cancer['feature_names'], rotation=90)
# plt.hlines(0, 0, cancer['data'].shape[1])
# plt.ylim(-5, 5)
# plt.xlabel('Feature')
# plt.ylabel('Coefficient mangnitude')
# plt.legend()
# plt.show()
# Comparing magnitude coefficients for each feature
# by predicted class with different values of C


# for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
#   lr_l1 = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)
#   print 'Training set accuracy of l1 logreg with C={:.3f}: {:.3f}'.format(C, lr_l1.score(X_train, y_train))
#   print 'Test set accuracy of l1 logreg with C={:.3f}: {:.3f}'.format(C, lr_l1.score(X_test, y_test))
#   plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))
# plt.xticks(range(cancer['data'].shape[1]), cancer['feature_names'], rotation=90)
# plt.hlines(0, 0, cancer['data'].shape[1])
# plt.xlabel('Features')
# plt.ylabel('Coefficient magnitudes')
# plt.ylim(-5, 5)
# plt.legend()
# plt.show()
# Plotting coefficients for LogisticRegression using L1 as the norm
# for regularization instead of L2
# Like for linear regression, this model only considers a few features
# depending on the strength of the regularization


X, y = make_blobs(random_state=42)
# mglearn.discrete_scatter(X[:,0], X[:,1], y)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.legend(['Class 0', 'Class 1', 'Class 2'])
# plt.show()
# Makes scatterplot of data points in a 2D plane
# which belong to one of 3 distinct classes


linear_svm = LinearSVC().fit(X, y)
# print 'Coefficient shape: ', linear_svm.coef_.shape
# print 'Intercept shape: ', linear_svm.intercept_.shape
# Coefficient shape is (3, 2), intercept shape is (3,)
# Coef elements are 2D vectors, one coef for each feature


# mglearn.discrete_scatter(X[:,0], X[:,1], y)
# line = np.linspace(-15, 15)
# for coef, intercept, color in zip(
#   linear_svm.coef_,
#   linear_svm.intercept_,
#   mglearn.cm3.colors,
# ):
#   plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.ylim(-10, 15)
# plt.xlim(-10, 8)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line Class 0', 'Line Class 1', 'Line Class 2'], loc=(1.01, 0.3))
# plt.show()
# Show plot with decision boundaries for each of the 3 classes


mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
line = np.linspace(-15, 15)
# for coef, intercept, color in zip(
#   linear_svm.coef_,
#   linear_svm.intercept_,
#   mglearn.cm3.colors,
# ):
#   plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line Class 0', 'Line Class 1', 'Line Class 2'], loc=(1.01, .3))
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.show()
# Plots the decision boundaries for the multi-class classifier
