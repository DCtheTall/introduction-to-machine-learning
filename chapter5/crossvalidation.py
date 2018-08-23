"""
Cross Validation
----------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, ShuffleSplit, GroupKFold
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression


iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target)
# print 'Cross-validation scores: {}'.format(scores)
# Cross validation separates the data into k evenly sized sets
# The model uses the union of k-1 sets as the training set and
# returns k accuracy scores, by default cross_val_score uses 3 sections


scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
# print 'Cross-validation scores: {}'.format(scores)
# Prints 5 scores
# print 'Average cross-validation score: {}'.format(scores.mean())
# It is also common to print the mean of the scores


# print 'Iris labels:\n{}'.format(iris.target)
# The iris dataset is split by class, so a normal cross-validation
# would not be effective


# mglearn.plots.plot_stratified_cross_validation()
# plt.show()
# sklearn uses stratified cross validation, which splits
# the data into datasets with even numbers of each class


kfold = KFold(n_splits=5)
# print 'Cross-validation score:\n{}'.format(
#   cross_val_score(logreg, iris.data, iris.target, cv=kfold))
# KFold uses standard k-fold cross-validation instead of stratified


kfold = KFold(n_splits=3)
# print 'Cross-validation score:\n{}'.format(
#   cross_val_score(logreg, iris.data, iris.target, cv=kfold))
# Here we see using standard k-fold is not good for the iris dataset


kfold = KFold(n_splits=3, shuffle=True, random_state=0)
# print 'Cross-validation score:\n{}'.format(
#   cross_val_score(logreg, iris.data, iris.target, cv=kfold))
# KFold allows you to shuffle the data for better results
# Setting random_state can allow the shuffling to be deterministic


loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
# print 'Number of cv iterations: {}'.format(len(scores))
# print 'Mean accuracy: {:.2f}'.format(scores.mean())
# LeaveOneOut works like KFold but it uses single samples for the partitions
# of the dataset


# mglearn.plots.plot_shuffle_split()
# plt.show()
# Shuffle-split cross-validation randomly selects points into a fixed
# sized training and test set


shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
# print 'Cross-validation scores: {}'.format(scores)
# Here ShuffleSplit split the dataset in half randomly 10 times
# into the test and training sets


X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
# print 'Cross-validation scores:\n{}'.format(scores)
# GroupKFold splits the dataset by given groups, which are useful for certain
# applications when you want your dataset to be able to generalize well
# to groups that are not in the training set


mglearn.plots.plot_group_kfold()
plt.show()
# GroupKFold divides the data into training and test sets by
# the groups so that we get an accurate picture of how the model
# generalizes
