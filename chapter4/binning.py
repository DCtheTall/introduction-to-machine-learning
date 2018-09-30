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
plt.plot(line, reg.predict(line), label="Decision tree")
reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="Linear regression")
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.legend(loc='Best')
plt.show()
# Comparing decision tree with linear regresssion
# Decision trees can model the data much more closely


bins = np.linspace(-3, 3, 11)
print 'bins: {}'.format(bins)
# Separate the linear space into 10 bins where
# the feature values range from -3 to -2.4, -2.4 to -1.8, etc


which_bin = np.digitize(X, bins=bins)
print '\nData points:\n{}'.format(X[:5])
print '\nBin membership:\n{}'.format(which_bin[:5])
# np.digitize encoded the continuous input feature
# into a discrete feature representing which bin the
# each point belongs to


encoder = OneHotEncoder(sparse=False).fit(which_bin)
X_binned = encoder.transform(which_bin)
print X_binned[:5]
print 'X_binned.shape: {}'.format(X_binned.shape)
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


X_combined = np.hstack([X, X_binned])
print X_combined.shape
# One way to improve the linear model after binning is to add
# slopes back to the bins using np.hstack


reg = LinearRegression().fit(X_combined, y)
line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='Linear regression combined')
for b in bins:
  plt.plot([b, b], [-3, 3], ':', c='k', linewidth=1)
plt.legend(loc='best')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.plot(X[:,0], y, 'o', c='k')
plt.show()
# The linear regression learned to separate the
# data into each bin and it learned a shared slope
# across all bins. This is less preferable than
# finding a separate slope across all bins


X_product = np.hstack([X_binned, X * X_binned])
print X_product.shape
# We can achieve separate slopes for each bin by
# taking a product of each bin indicator with
# the original input feature
# Each point has 20 features now, 10 indicate
# which bin its in and the other 10 are the corresponding
# products with the original input feature


reg = LinearRegression().fit(X_product, y)
line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')
for b in bins:
  plt.plot([b, b], [-3, 3], ':', c='k', linewidth=1)
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.legend(loc='best')
plt.show()
# This time, each bin has its own slope
