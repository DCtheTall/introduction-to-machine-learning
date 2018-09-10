"""
Uncertainty Estimates from Classifiers
--------------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X, y = make_circles(noise=.25, factor=.5, random_state=1)
y_named = np.array(['blue', 'red'])[y]
(
  X_train,
  X_test,
  y_train_named,
  y_test_named,
  y_train,
  y_test
) = train_test_split(X, y_named, y, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train_named)
# print 'X_test.shape: {}'.format(X_test.shape)
# print 'Decision function shape: {}'.format(gbrt.decision_function(X_test).shape)
# The shape of the test data compared to the shape of the return
# value of the decision function
# Each element in the decision function output is the predicted value of
# each item in the test set, the sign of the number indicates which class
# the model believes the item to be in.


# print 'Decision function:\n{}'.format(gbrt.decision_function(X_test)[:6])
# First 6 elements in the return value of the decision fn


# print 'Thresholded decision function:\n{}'.format(
#     gbrt.decision_function(X_test) > 0)
# print 'Predictions:\n{}'.format(
#     gbrt.predict(X_test))
# Decision fn result > 0 implies that its in class 0, less than 0 implies class 1


# decision_fn = gbrt.decision_function(X_test)
# print 'Decision function minimum: {:.2f} maximum: {:.2f}'.format(np.min(decision_fn), np.max(decision_fn))
# Arbitrary range of the decision function can make it hard to interpret


# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
# scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)
# for ax in axes:
#   mglearn.discrete_scatter(
#     X_test[:,0], X_test[:,1], y_test, markers='^', ax=ax)
#   mglearn.discrete_scatter(
#     X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
#   ax.set_xlabel('Feature 0')
#   ax.set_ylabel('Feature 1')
# cbar = plt.colorbar(scores_image, ax=axes.tolist())
# axes[0].legend([
#   'Test class 0',
#   'Test class 1',
#   'Train class 0',
#   'Train class 1',
# ], ncol=4, loc=(.1, 1.1))
# plt.show()
# Plots the decision boundary of the gradient boosted classifier
# and the values of the decision function. This visualization may
# provide more information but it is hard to discern classes


# print 'Shape of probabilities: {}'.format(gbrt.predict_proba(X_test).shape)
# Print shape of the probability prediction result


# print 'Predicted probabilities:\n{}'.format(gbrt.predict_proba(X_test)[:6])
# Print the probabilities the model predicts that each point belongs to either class


# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# mglearn.tools.plot_2d_separator(
#   gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
# scores_image = mglearn.tools.plot_2d_scores(
#   gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')
# for ax in axes:
#   mglearn.discrete_scatter(
#     X_test[:,0], X_test[:,1], y_test, markers='^', ax=ax)
#   mglearn.discrete_scatter(
#     X_train[:,0], X_train[:,1], y_train, markers='o', ax=ax)
#   ax.set_xlabel('Feature 0')
#   ax.set_ylabel('Feature 1')
# cbar = plt.colorbar(scores_image, ax=axes.tolist())
# axes[0].legend(['Test class 0', 'Test class 1', 'Train class 0', 'Train class 1'], ncol=4, loc=(.1, 1.1))
# plt.show()
# Plot of the decision boundary and the probability of each class. Here it is easier
# to visualize where the uncertainty in the model is


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
  iris.data, iris.target, random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0).fit(X_train, y_train)
# print 'Decision function shape: {}'.format(gbrt.decision_function(X_test).shape)
# print 'Decision function:\n{}'.format(gbrt.decision_function(X_test)[:6,:])
# Printing the shape and the values of the decision function for a multiclass
# gradient boosted classifier


# print 'Argmax of decision function:\n{}'.format(
#   np.argmax(gbrt.decision_function(X_test), axis=1))
# print 'Predictions:\n{}'.format(gbrt.predict(X_test))
# The predicted class is the maximum entry in each data point returned from the decision fn


# print 'Predicted probabilities:\n{}'.format(
#   gbrt.predict_proba(X_test)[:6])
# print 'Sums: {}'.format(gbrt.predict_proba(X_test)[:6].sum(axis=1))
# Probabilites for a multiclass classifier, each of the probabilites for each data point sum to 1


# print 'Argmax of predicted probabilities:\n{}'.format(
#   np.argmax(gbrt.predict_proba(X_test), axis=1))
# print 'Predictions:\n{}'.format(gbrt.predict(X_test))
# The maximum probability of each data point reveals which class the classifier will predict it
# to be in


logreg = LogisticRegression()
named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print 'Unique classes in training data: {}'.format(logreg.classes_)
print 'Predictions: {}'.format(logreg.predict(X_test)[:10])
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print 'Argmax of decision function: {}'.format(argmax_dec_func[:10])
print 'Argmax combined with classes_: {}'.format(
  logreg.classes_[argmax_dec_func][:10])
# When using string names for classes, you have to combine the decision fn or predict_proba
# with model.classes_ to have it reflect the prediction result
