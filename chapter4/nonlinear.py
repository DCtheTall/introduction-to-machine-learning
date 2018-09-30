"""
Nonlinear transformations
-------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)
X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
print 'Number of feature appearances:\n{}'.format(np.bincount(X[:, 0]))
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='k')
plt.ylabel('Number of appearances')
plt.xlabel('Value')
plt.show()
# Synthetic dataset of a positive integer, the plot above shows
# how many times each value for the feature appears in the set


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print 'Test score: {:.3f}'.format(score)
# Score is .622, not very good


X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel('Number of appearances')
plt.xlabel('Value')
plt.show()
# Plot of the data after applying a logarithmic transformation


score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print 'Test score: {:.3f}'.format(score)
# New score is .875, much better than before the transformation

