"""
Lasso regression
----------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from ridgeregress import ridge01


# On theory: https://en.wikipedia.org/wiki/Lasso_(statistics)#Basic_form


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


lasso = Lasso().fit(X_train, y_train)
# print 'Training score: {:.2f}'.format(lasso.score(X_train, y_train)) # .29
# print 'Test score: {:.2f}'.format(lasso.score(X_test, y_test)) # .21
# print 'Number of features used: {}'.format(np.sum(lasso.coef_ != 0))
# Only uses 4 features, underfitting


lasso001 = Lasso(alpha=0.01, max_iter=1e5).fit(X_train, y_train)
# print 'Training score: {:.2f}'.format(lasso001.score(X_train, y_train)) # .9
# print 'Test score: {:.2f}'.format(lasso001.score(X_test, y_test)) # .77
# print 'Number of features used: {}'.format(np.sum(lasso001.coef_ != 0))
# Much better fit, reducing the size of alpha makes the feature
# selection less strict

lasso00001 = Lasso(alpha=0.0001, max_iter=1e5).fit(X_train, y_train)
# print 'Training score: {:.2f}'.format(lasso00001.score(X_train, y_train)) # .95
# print 'Test score: {:.2f}'.format(lasso00001.score(X_test, y_test)) # .64
# print 'Number of features used: {}'.format(np.sum(lasso00001.coef_ != 0))
# Example of overfitting by setting alpha too low, this becomes to
# close to regular OLS


plt.plot(lasso.coef_, 's', label='Lasso alpha=1')
plt.plot(lasso001.coef_, '^', label='Lasso alpha=0.01')
plt.plot(lasso00001.coef_, 'v', label='Lasso alpha=0.0001')
plt.plot(ridge01.coef_, 'o', label='Ridge alpha=0.1')
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.show()
