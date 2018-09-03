"""
Evalutation Metrics and Scoring
-------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report


digits = load_digits()
y = digits['target'] == 9
X_train, X_test, y_train, y_test = \
  train_test_split(digits['data'], y, random_state=0)
# Binomial data that has a 9:1 False-to-True ratio


dummy_majority = \
  DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
# print 'Unique predicted labels: {}'.format(np.unique(pred_most_frequent))
# print 'Test score: {:.2f}'.format(dummy_majority.score(X_test, y_test))
# Since the data is imbalanced 9:1, a dummy classifier can achieve
# 90% accuracy by just predicting the majority class all the time.
# This illustrates that accuracy is not always a good metric


tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
# print 'Test score: {:.2f}'.format(tree.score(X_test, y_test))
# 92% accuracy, which is only slightly better than just predicting
# the most frequent class


dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
# print 'Dummy score: {:.2f}'.format(dummy.score(X_test, y_test))
# You can achieve ~80% accuracy with just random guessing


logreg = LogisticRegression(C=.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
# print 'Logreg score: {:.2f}'.format(logreg.score(X_test, y_test))
# Logistic Regression does the best so far with 98% accuracy,
# but since random guessing achieves 80%, it's hard to tell if this
# is helpful.


confusion = confusion_matrix(y_test, pred_logreg)
# print 'Confusion matrix:\n{}'.format(confusion)
# Prints a 2x2 confusion matrix which represents
# the times the classifier predicted the positive
# and negative classes correctly (TP & TN),
# as well as how many false negatives and false
# positives there were (FN & FP).
# The matrix is structured like:
#   TN FP
#   FN TP


# mglearn.plots.plot_confusion_matrix_illustration()
# mglearn.plots.plot_binary_confusion_matrix()
# plt.show()
# Plots illustration of the confusion matrix structure
# described above


# print 'Most frequent:\n{}\n'.format(
#   confusion_matrix(y_test, pred_most_frequent))
# print 'Dummy model:\n{}\n'.format(
#   confusion_matrix(y_test, pred_dummy))
# print 'Decision tree:\n{}\n'.format(
#   confusion_matrix(y_test, pred_tree))
# print 'Logistic regression:\n{}\n'.format(
#   confusion_matrix(y_test, pred_logreg))
# Comparing the confusion matrices of the different
# classifiers above


# Accuracy from confusion matrix:
#   (TP + TN) / (TP + TN + FP + FN)

# Precision: (how many real positives over the sum of all positive predictions)
#   TP / (TP + FP)
# Precision is good for testing for limiting false positives

# Recall: (how many positive samples were captured over total number of positive samples)
#   TP / (TP + FN)
# Recall is a better metric for when it's important to capture all positive samples


# For more info visit: https://en.wikipedia.org/wiki/Sensitivity_and_specificity


model_names = [
  'most frequent',
  'dummy',
  'tree',
  'logistic regression',
]
pred_list = [
  pred_most_frequent,
  pred_dummy,
  pred_tree,
  pred_logreg,
]
# for name, pred in zip(model_names, pred_list):
#   print 'f1 score {}: {:.2f}'.format(name, f1_score(y_test, pred))
# A helpful metric is the f1 score, the harmonic mean of the
# recall and precision.
# It seems to capture what makes a model a better predictor
# but it is harder to interpret than accuracy


def print_classification_report(pred):
  print classification_report(
    y_test, pred, target_names=['not nine', 'nine'])
# print_classification_report(pred_most_frequent)
# print_classification_report(pred_dummy)
# print_classification_report(pred_logreg)
# The classification report shows what the precision, recal,
# and f1 score of the model when each class in the samples
# is treated as the positive class.
# It also shows the average result.



