"""
Naive Bayes Classifiers
-----------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB


X = np.array([
  [0, 1, 0, 1],
  [1, 0, 1, 1],
  [0, 0, 0, 1],
  [1, 0, 1, 0],
])
y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y):
  counts[label] = X[y == label].sum(axis=0)
print 'Feature counts:\n{}'.format(counts)
# Example in the book to show how
# BernoulliMB aggregates features


X = np.random.randint(2, size=(1000, 1000))
y = np.random.randint(5, size=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = BernoulliNB(alpha=100.).fit(X_train, y_train)
print 'Training set score: {:.3f}'.format(clf.score(X_train, y_train))
print 'Test set score: {:.3f}'.format(clf.score(X_test, y_test))
# BernoulliNB classifies a set of data points with vectors of binary features (0 or 1)


X = np.random.randint(5, size=(1000, 1000))
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = MultinomialNB(alpha=1000.).fit(X_train, y_train)
print 'Training set score: {:.3f}'.format(clf.score(X_train, y_train))
print 'Test set score: {:.3f}'.format(clf.score(X_test, y_test))
# MultinomialNB classifies a set of discrete data points with vectors of discrete features


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([0, 0, 0, 1, 1, 1])
clf = GaussianNB().fit(X, y)
print clf.predict([[-0.8, -1]])
# GaussianNB can classify data with real-valued feature vectors
