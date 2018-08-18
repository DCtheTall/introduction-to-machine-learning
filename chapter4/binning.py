"""
Binning
-------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder


X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
# plt.plot(line, reg.predict(line), label="Decision tree")
reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), label="Linear regression")
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel('Regression output')
# plt.xlabel('Input feature')
# plt.legend(loc='Best')
# plt.show()
# Comparing decision tree with linear regresssion
# Decision trees can model the data much more closely


bins = np.linspace(-3, 3, 11)
# print 'bins: {}'.format(bins)
# Separate the linear space into 10 bins where
# the feature values range from -3 to -2.4, -2.4 to -1.8, etc


which_bin = np.digitize(X, bins=bins)
# print '\nData points:\n{}'.format(X[:5])
# print '\nBin membership:\n{}'.format(which_bin[:5])
# np.digitize encoded the continuous input feature
# into a discrete feature representing which bin the
# each point belongs to


encoder = OneHotEncoder(sparse=False).fit(which_bin)
X_binned = encoder.transform(which_bin)
# print X_binned[:5]
# print 'X_binned.shape: {}'.format(X_binned.shape)
# One hot encoder does the same thing as pandas.get_dummies (see categoricalvars.py)
# except it only works on integer categorical variables
# It separates the categorical feature into multiple binary features


line_binned = encoder.transform(np.digitize(line, bins=bins))
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='Linear regression binned')
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='Decision tree binned')
plt.plot(X[:,0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc='best')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.show()
# Now the two lines are right on top of each other
# Binning data seems to help linear models make better predictions
# but it made the decision tree perform worse
# Decision trees in general can learn what bins to put data in
# based on multiple features
