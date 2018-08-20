"""
Automatic Feature Selection
---------------------------

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Select deterministic random numbers
cancer = load_breast_cancer()
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# Add noise features to the cancer data
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(
  X_w_noise, cancer.target, random_state=0, test_size=.5)
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
# print 'X_train.shape: {}'.format(X_train.shape)
# print 'X_train_selected.shape: {}'.format(X_train_selected.shape)
# Only 50% of the features were selected by the SelectPercentile transform


mask = select.get_support()
# print mask
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel('Sample index')
# plt.yticks(())
# plt.show()
# The selection process was able to recover most of the original
# features and filter most of the noise


lr = LogisticRegression().fit(X_train, y_train)
# print 'Score with all features: {:.3f}'.format(lr.score(X_test, y_test))
# Score with all features is .93
lr.fit(X_train_selected, y_train)
X_test_selected = select.transform(X_test)
# print 'Score with selected features: {:.3f}'.format(lr.score(X_test_selected, y_test))
# Score with selected features is .94
# A slightly better performance


select = SelectFromModel(
  RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
# print 'X_train.shape: {}'.format(X_train.shape)
# print 'X_train_l1.shape: {}'.format(X_train_l1.shape)
# Here the select transform removed half of the features
# RandomForestClassifier determined the importance of each features
# and removed features less important than th median


mask = select.get_support()
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel('Sample index')
# plt.yticks(())
# plt.show()
# Here we see all but 2 of the original features were selected


X_test_li = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_li, y_test)
# print 'Test set score: {:.3f}'.format(score)
# This time the score is .951, which is a better boost in performance
# than SelectPercentile


select = RFE(RandomForestClassifier(
  n_estimators=100, random_state=42), n_features_to_select=40)
select.fit(X_train, y_train)
mask = select.get_support()
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel('Feature index')
# plt.yticks(())
# plt.show()
# Plotting which features were selected by the Recursive Feature Elimination (RFE)
# this method of feature selection is more computationally expensive than
# the others since it has to iteratively train a model


X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
# print 'Test score: {:.3f}'.format(score)
# Scores .951
# print 'Test score: {:.3f}'.format(select.score(X_test, y_test))
# Also scores .951, after thie feature selection the linear model
# performs just as well as the random forest
