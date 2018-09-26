"""
Ridge regression
----------------

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from linregress import lr


# More on theory: https://en.wikipedia.org/wiki/Tikhonov_regularization


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
ridge = Ridge().fit(X_train, y_train)
print 'Train score: {:.2f}'.format(ridge.score(X_train, y_train)) # .89
print 'Test score: {:.2f}'.format(ridge.score(X_test, y_test)) # .75
# Ridge intentionally dampens the effect of each attribute on the
# prediction to avoid overfitting


ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print 'Train test score: {:.2f}'.format(ridge10.score(X_train, y_train)) # .79
print 'Test test score: {:.2f}'.format(ridge10.score(X_test, y_test)) # .64
# You can add a regularization term (alpha) to fit to your particular model
# In this case alpha=10 doesn't help


ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print 'Train test score: {:.2f}'.format(ridge01.score(X_train, y_train))  # .93
print 'Test test score: {:.2f}'.format(ridge01.score(X_test, y_test))  # .77
# It appears lowering the alpha to 0.1 was useful here


# plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
# plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
# plt.plot(ridge01.coef_, 'v', label='Ridge alpha=0.1')
# plt.plot(lr.coef_, 'o', label='Linear regression')
# plt.xlabel('Coefficient index')
# plt.ylabel('Coefficient magnitude')
# plt.hlines(0, 0, len(lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()
# Plot shows how larger alphas restrict the coefficient
# for linear regression to a tighter bound than smaller
# alphas (lin regress -> alpha = 0)


# mglearn.plots.plot_ridge_n_samples()
# plt.show()
# Comparing test and training accuracy of linear regression to ridge regression
