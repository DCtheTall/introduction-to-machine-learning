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
from sklearn.metrics import (
  confusion_matrix,
  f1_score,
  classification_report,
  precision_recall_curve,
  average_precision_score,
  roc_curve,
  roc_auc_score,
)
from mglearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


digits = load_digits()
y = digits['target'] == 9
X_train, X_test, y_train, y_test = \
  train_test_split(digits['data'], y, random_state=0)
# Binomial data that has a 9:1 False-to-True ratio


dummy_majority = \
  DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print 'Unique predicted labels: {}'.format(np.unique(pred_most_frequent))
print 'Test score: {:.2f}'.format(dummy_majority.score(X_test, y_test))
# Since the data is imbalanced 9:1, a dummy classifier can achieve
# 90% accuracy by just predicting the majority class all the time.
# This illustrates that accuracy is not always a good metric


tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print 'Test score: {:.2f}'.format(tree.score(X_test, y_test))
# 92% accuracy, which is only slightly better than just predicting
# the most frequent class


dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print 'Dummy score: {:.2f}'.format(dummy.score(X_test, y_test))
# You can achieve ~80% accuracy with just random guessing


logreg = LogisticRegression(C=.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print 'Logreg score: {:.2f}'.format(logreg.score(X_test, y_test))
# Logistic Regression does the best so far with 98% accuracy,
# but since random guessing achieves 80%, it's hard to tell if this
# is helpful.


confusion = confusion_matrix(y_test, pred_logreg)
print 'Confusion matrix:\n{}'.format(confusion)
# Prints a 2x2 confusion matrix which represents
# the times the classifier predicted the positive
# and negative classes correctly (TP & TN),
# as well as how many false negatives and false
# positives there were (FN & FP).
# The matrix is structured like:
#   TN FP
#   FN TP


mglearn.plots.plot_confusion_matrix_illustration()
mglearn.plots.plot_binary_confusion_matrix()
plt.show()
# Plots illustration of the confusion matrix structure
# described above


print 'Most frequent:\n{}\n'.format(
  confusion_matrix(y_test, pred_most_frequent))
print 'Dummy model:\n{}\n'.format(
  confusion_matrix(y_test, pred_dummy))
print 'Decision tree:\n{}\n'.format(
  confusion_matrix(y_test, pred_tree))
print 'Logistic regression:\n{}\n'.format(
  confusion_matrix(y_test, pred_logreg))
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
for name, pred in zip(model_names, pred_list):
  print 'f1 score {}: {:.2f}'.format(name, f1_score(y_test, pred))
# A helpful metric is the f1 score, the harmonic mean of the
# recall and precision.
# It seems to capture what makes a model a better predictor
# but it is harder to interpret than accuracy


def print_classification_report(pred):
  print classification_report(
    y_test, pred, target_names=['not nine', 'nine'])
print_classification_report(pred_most_frequent)
print_classification_report(pred_dummy)
print_classification_report(pred_logreg)
# The classification report shows what the precision, recal,
# and f1 score of the model when each class in the samples
# is treated as the positive class.
# It also shows the average result.


X, y = make_blobs(
  n_samples=(400, 50), centers=2, cluster_std=[7, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
mglearn.plots.plot_decision_threshold()
plt.show()
# Plots how changing the decision threshold can affect the outcome of a
# classifier


print(classification_report(y_test, svc.predict(X_test)))
# As we can see from the classification report, predicting
# class 1 has a low precision and low recall, let's say
# we want to improve this by changing the decision threshold


y_pred_lower_threshold = svc.decision_function(X_test) > -.8
print classification_report(y_test, y_pred_lower_threshold)
# Here the recall of predicting class 1 went up. This makes
# sense, since decreasing the threshold will make the model
# more likely to predict that a data point is in class 1,
# even if it is a false positive.


precision, recall, thresholds = \
  precision_recall_curve(y_test, svc.decision_function(X_test))
# Precision recall curve fn returns a list of prediction and
# recall values for all possible thresholds (all values that appear
# in the decision fn) in sorted order


X, y = make_blobs(
  n_samples=(4000, 500),
  centers=2,
  cluster_std=[7, 2],
  random_state=22,
)
X_train, X_test, y_train, y_test = \
  train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = \
  precision_recall_curve(y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(thresholds))
plt.plot(
  precision[close_zero],
  recall[close_zero],
  'o',
  markersize=10,
  label='threshold zero',
  fillstyle='none',
  c='k',
  mew=2,
)
plt.plot(precision, recall, label='precision recall curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc='best')
plt.show()
# Plot the precision vs recall function and then find the threshold
# closest to zero. The closer the curve gets to the top right of the
# plane, the better. It seems as though we must trade off between
# precision and recall, which makes sense. Models with high recall
# may be generating a lot of false positives (high recall), whereas
# models with a high threshold may produce false negatives


rf = RandomForestClassifier(
  n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)
precision_rf, recall_rf, thresholds_rf = \
  precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(precision, recall, label='svc')
plt.plot(
  precision[close_zero],
  recall[close_zero],
  'o',
  markersize=10,
  label='threshold zero',
  fillstyle='none',
  c='k',
  mew=2,
)
plt.plot(precision_rf, recall_rf, label='rf')
close_default_rf = np.argmin(np.abs(thresholds_rf - .5))
plt.plot(
  precision_rf[close_default_rf],
  recall_rf[close_default_rf],
  '^',
  c='k',
  markersize=10,
  label='threshold 0.5 rf',
  fillstyle='none',
  mew=2,
)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.legend(loc='best')
plt.show()
# From the curves we see that RandomForest performs better
# around the extremes but around the middle, SVC performs
# better.


print 'f1_score of random forest: {:.3f}'.format(f1_score(y_test, rf.predict(X_test)))
print 'f1_score of svc: {:.3f}'.format(f1_score(y_test, svc.predict(X_test)))
# Looking at only the f1 score only gives a general idea of
# model performance. If we had not looked at the plots above
# we would have missed the subtleties


ap_rf = average_precision_score(
  y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(
  y_test, svc.decision_function(X_test))
print 'Average precision of random forest: {:.3f}'.format(ap_rf)
print 'Average precision of svc: {:.3f}'.format(ap_svc)
# average_precision score averages the precision over the possible
# thresholds, aka the area under the precision-recall curve (from
# the y-axis). Since the curve is a function that goes from 0 to 1,
# the average precision is always from 0 to 1.


fpr, tpr, thresholds = roc_curve(
  y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR (recall)')
close_zero = np.argmin(np.abs(thresholds))
plt.plot(
  fpr[close_zero],
  tpr[close_zero],
  'o',
  markersize=10,
  label='threshold zero',
  fillstyle='none',
  c='k',
  mew=2,
)
plt.legend(loc=4)
plt.show()
# This code plots the Receiver Operating Characteristics (ROC)
# curve, which plots FPR (false positive rate) on the x-axis
# against TPR (true positive rate, same as recall) on the y-axis.
# For this curve, we want the threshold that leads to the point
# closest to the top left. In this graph, we see the default
# threshold is not the best performance we can get.

# FPR = FP / (FP + TN)
# TPR = TP / (TP + FN) = recall


fpr_rf, tpr_rf, thresholds_rf = \
  roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='ROC Curve SVC')
plt.plot(fpr_rf, tpr_rf, label='ROC Curve RF')
plt.xlabel('FPR')
plt.ylabel('TPR (recall)')
plt.plot(
  fpr[close_zero],
  tpr[close_zero],
  'o',
  markersize=10,
  label='threshold zero SVC',
  fillstyle='none',
  c='k',
  mew=2,
)
close_default_rf = np.argmin(np.abs(thresholds_rf - .5))
plt.plot(
  fpr_rf[close_default_rf],
  tpr_rf[close_default_rf],
  '^',
  markersize=10,
  label='threshold 0.5 RF',
  fillstyle='none',
  c='k',
  mew=2,
)
plt.legend(loc=4)
plt.show()
# Here we plot the ROC curves for the SVC and the Random Forest


rf_auc = roc_auc_score(
  y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(
  y_test, svc.decision_function(X_test))
print 'AUC for Random Forest: {:.3f}'.format(rf_auc)
print 'AUC for SVC: {:.3f}'.format(svc_auc)
# A good way to summarize the ROC curve with a single
# number is to compute the area under the curve (AUC)
# Here we see the RandomForest performs a bit better than the SVC


y = digits['target'] == 9
X_train, X_test, y_train, y_test = \
  train_test_split(digits['data'], y, random_state=0)
plt.figure()
for gamma in [1, 0.05, 0.01]:
  svc = SVC(gamma=gamma).fit(X_train, y_train)
  accuracy = svc.score(X_test, y_test)
  auc = roc_auc_score(y_test, svc.decision_function(X_test))
  fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
  print 'gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}'.format(
    gamma, accuracy, auc)
  plt.plot(fpr, tpr, label='gamma={:.3f}'.format(gamma))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc='best')
plt.show()
# Plotting the ROC curve for different values of gamma while using
# SVC to predict whether a handwritten digit is a 9 or not.
# Here we see the accuracy is the same for all settings, but the ROC
# curves and AUC score give us an idea of what models perform better
