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
  TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline


POWERS_OF_TEN = [10 ** i for i in range(-3, 3)]


reviews_train = load_files('./aclImdb/train/')
text_train, y_train = reviews_train.data, reviews_train.target
text_train, y_train = \
    zip(*filter(lambda (X, y): y < 2, zip(text_train, y_train)))
# Upon inspection I noticed this data has 3 classes in the training
# set, which does not match the book example.
# I applied my own data transformation on it to make it match
# the example.
print 'Type of text_train: {}'.format(type(text_train))
print 'Length of text_train: {}'.format(len(text_train))
print 'text_train[6]:\n{}'.format(text_train[6])
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
print 'Number of documents in the test data: {}'.format(len(text_test))
print 'Samples per class (test): {}'.format(np.bincount(y_test))
# The test set is the same size/structure, 12500 of each class


vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print 'X_train:\n{}'.format(repr(X_train))
# Data is represented by a 25000 x  74,849 matrix


feature_names = vect.get_feature_names()
print 'Number of features: {}'.format(len(feature_names))
print 'First 20 features:\n{}'.format(feature_names[:20])
print 'Features 20010 to 20030:\n{}'.format(feature_names[20010:20030])
print 'Every 2000th feature:\n{}'.format(feature_names[::2000])
# Printing some of the features to examine the dataset
# we just created with this preprocessing step


scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print 'Mean cross-validation accuracy: {:.2f}'.format(np.mean(scores))
# Already able to use Logistic Regression to classify negatives
# as positive or negative with 88% accuracy using cross validation


param_grid = {'C': POWERS_OF_TEN}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print 'Best cross validation score: {:.2f}'.format(grid.best_score_)
print 'Best parameters: {}'.format(grid.best_params_)
# Able to tune the parameters to find that we can get 89%
# accuracy when C (regularization constant) is set to 0.1


X_test = vect.transform(text_test)
print 'Test score: {:.2f}'.format(grid.score(X_test, y_test))
# The current model is also able to generalize with 88%
# accuracy as well


vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
feature_names = vect.get_feature_names()
print 'X_train with min_df: {}'.format(repr(X_train))
# 25000 x 27271 matrix, features have been reduced
print 'First 50 features:\n{}'.format(feature_names[:50])
print 'Features 20010 to 20030:\n{}'.format(feature_names[20010:20030])
print 'Every 700th feature:\n{}'.format(feature_names[::700])
# We can set the minimum number of documents a term must show
# up in with the min_df setting to reduce features which do not
# help us


grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
# print 'Best cross-validation score: {:.2f}'.format(grid.best_score_)
# This feature reduction had no noticeable effect on the cross validation


print 'Number of stop words: {}'.format(len(ENGLISH_STOP_WORDS))
print 'Every 10th stop word:\n{}'.format(list(ENGLISH_STOP_WORDS)[::10])


vect = CountVectorizer(min_df=5, stop_words='english').fit(text_train)
X_train = vect.transform(text_train)
print 'X_train with stop words:\n{}'.format(repr(X_train))
# 25000 x 26966 matrix, an additional 305 more features were removed
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print 'Best cross-validation score: {:.2f}'.format(grid.best_score_)
# Still only 88% accurate


pipe = make_pipeline(
  TfidfVectorizer(min_df=5),
  LogisticRegression(),
)
param_grid = {'logisticregression__C': POWERS_OF_TEN}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print 'Best cross-validation score: {:.2f}'.format(grid.best_score_)
# Here we used Term Frequency * Inverse Document (tfidf) Frequency as the value
# of each component of the term vectors
# The calculation is:
#
#  tfidf(w) = tf(w) * log((N + 1) / (N_w + 1)) + 1
#
# where N is the number of documents in the dataset and N_w is the number of docs
# with word w in the doc
# 89% Accuracy, no noticeable effect


vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
X_train = vectorizer.transform(text_train)
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
feature_names = np.array(vectorizer.get_feature_names())
print 'Features with lowest tfidf:\n{}'.format(
  feature_names[sorted_by_tfidf[:20]])
print 'Features with highest tfidf:\n{}'.format(
  feature_names[sorted_by_tfidf[-20:]])
sorted_by_idf = np.argsort(vectorizer.idf_)
print 'Features with lowest idf:\n{}'.format(
  feature_names[sorted_by_idf[:100]])
# Sorting my tfidf to see what words have high/low tfidf scores
# The lowest terms show up either less frequently or are used in many documents
# The high terms show up frequently but not in very many documents
# We also sort by Inverse Document Frequency (idf) to see terms with
# very low idf, what we find are common English words used in many
# types of writing


mglearn.tools.visualize_coefficients(
  grid.best_estimator_.named_steps['logisticregression'].coef_,
  feature_names,
  n_top_features=40,
)
plt.show()
# Terms that have most impact are words that express highly
# subjective opinions (i.e. "best", "worst")


pipe = make_pipeline(
  TfidfVectorizer(min_df=5),
  LogisticRegression(),
)
param_grid = {
  'logisticregression__C': POWERS_OF_TEN,
  'tfidfvectorizer__ngram_range': [(1, e) for e in range(1, 4)]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print 'Best cross-validation score: {:.2f}'.format(grid.best_score_)
print 'Best parameters:\n{}'.format(grid.best_params_)
# Running a grid search to see which regularization constant, C, and the
# largest n-gram size that gets the best results

scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
heatmap = mglearn.tools.heatmap(
  scores,
  xlabel='C',
  ylabel='ngram_range',
  cmap='viridis',
  fmt='%.3f',
  xticklabels=param_grid['logisticregression__C'],
  yticklabels=param_grid['tfidfvectorizer__ngram_range'],
)
plt.colorbar(heatmap)
plt.show()
# Heat map of the cross validation grid showing the scores
# with each parameter value


vect = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
plt.show()
# Plots the 40 most important features and their coefficients in the
# logistic regression


mask = np.array([len(feature.split(' ')) for feature in feature_names]) == 3
mglearn.tools.visualize_coefficients(coef.ravel()[mask], feature_names[mask], n_top_features=40)
plt.show()
# Shows the top 40 trigrams which the logistic regression
# model uses
