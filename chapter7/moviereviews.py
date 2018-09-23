"""
Sentiment Analysis of Movie Reviews
-----------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import (
  CountVectorizer,
  ENGLISH_STOP_WORDS,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV


POWERS_OF_TEN = [10 ** i for i in range(-3, 3)]


reviews_train = load_files('./aclImdb/train/')
text_train, y_train = reviews_train.data, reviews_train.target
text_train, y_train = \
    zip(*filter(lambda (X, y): y < 2, zip(text_train, y_train)))
# Upon inspection I noticed this data has 3 classes in the training
# set, which does not match the book example.
# I applied my own data transformation on it to make it match
# the example.
# print 'Type of text_train: {}'.format(type(text_train))
# print 'Length of text_train: {}'.format(len(text_train))
# print 'text_train[6]:\n{}'.format(text_train[6])
# Download movie review data collected by Stanford University
# and load training data


repl_br_tags = lambda text: [doc.replace(b'<br />', b' ') for doc in text]
text_train = repl_br_tags(text_train)
# Manual inspection of the data shows there are still some
# HTML <br /> tags left in the text.


# print 'Samples per class (training): {}'.format(np.bincount(y_train))
# There are 12500 samples of each class, 2 classes


reviews_test = load_files('./aclImdb/test/')
text_test, y_test = reviews_test.data, reviews_test.target
text_test = repl_br_tags(text_test)
# print 'Number of documents in the test data: {}'.format(len(text_test))
# print 'Samples per class (test): {}'.format(np.bincount(y_test))
# The test set is the same size/structure, 12500 of each class


vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
# print 'X_train:\n{}'.format(repr(X_train))
# Data is represented by a 25000 x  74,849 matrix


feature_names = vect.get_feature_names()
# print 'Number of features: {}'.format(len(feature_names))
# print 'First 20 features:\n{}'.format(feature_names[:20])
# print 'Features 20010 to 20030:\n{}'.format(feature_names[20010:20030])
# print 'Every 2000th feature:\n{}'.format(feature_names[::2000])
# Printing some of the features to examine the dataset
# we just created with this preprocessing step


# scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
# print 'Mean cross-validation accuracy: {:.2f}'.format(np.mean(scores))
# Already able to use Logistic Regression to classify negatives
# as positive or negative with 88% accuracy using cross validation


param_grid = {'C': POWERS_OF_TEN}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid.fit(X_train, y_train)
# print 'Best cross validation score: {:.2f}'.format(grid.best_score_)
# print 'Best parameters: {}'.format(grid.best_params_)
# Able to tune the parameters to find that we can get 89%
# accuracy when C (regularization constant) is set to 0.1


X_test = vect.transform(text_test)
# print 'Test score: {:.2f}'.format(grid.score(X_test, y_test))
# The current model is also able to generalize with 88%
# accuracy as well


vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
# feature_names = vect.get_feature_names()
print 'X_train with min_df: {}'.format(repr(X_train))
# print 'First 50 features:\n{}'.format(feature_names[:50])
# print 'Features 20010 to 20030:\n{}'.format(feature_names[20010:20030])
# print 'Every 700th feature:\n{}'.format(feature_names[::700])
# We can set the minimum number of documents a term must show
# up in with the min_df setting to reduce features which do not
# help us


grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid.fit(X_train, y_train)
# print 'Best cross-validation score: {:.2f}'.format(grid.best_score_)
# This feature reduction had no noticeable effect on the cross validation


# print 'Number of stop words: {}'.format(len(ENGLISH_STOP_WORDS))
# print 'Every 10th stop word:\n{}'.format(list(ENGLISH_STOP_WORDS)[::10])


vect = CountVectorizer(min_df=5, stop_words='english').fit(text_train)
X_train = vect.transform(text_train)
print 'X_train with stop words:\n{}'.format(repr(X_train))
#
