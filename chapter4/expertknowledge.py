"""
Utilizing Expert Knowledge
--------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


citibike = mglearn.datasets.load_citibike()
# print 'Citi Bike data:\n{}'.format(citibike.head())
# Loading publicly available Citi Bike data


# plt.figure(figsize=(10, 3))
xticks = pd.date_range(
  start=citibike.index.min(), end=citibike.index.max(), freq='D')
# plt.xticks(xticks, xticks.strftime('%a %m-%d'), rotation=90, ha='left')
# plt.plot(citibike, linewidth=1)
# plt.xlabel('Date')
# plt.ylabel('Rentals')
# plt.show()
# Plot of Citi Bike rentals over time during August 2015


X = citibike.index.astype('int64').values.reshape(-1, 1) // (10 ** 9)
y = citibike.values
# We will use the POSIX time as our input feature and the number
# of rentals as the output feature

n_train = 184

def eval_on_features(features, target, regressor):
  """
  Function to evaluate and plot a regressor for a given feature set

  """
  X_train, X_test = features[:n_train], features[n_train:]
  y_train, y_test = target[:n_train], target[n_train:]
  regressor.fit(X_train, y_train)
  print 'Test set R^2: {:.2f}'.format(regressor.score(X_test, y_test))
  y_pred = regressor.predict(X_test)
  y_pred_train = regressor.predict(X_train)
  plt.figure(figsize=(10, 3))
  plt.xticks(
    range(0, len(X), 8), xticks.strftime('%a %m-%d'), rotation=90, ha='left')
  plt.plot(range(n_train), y_train, label='train')
  plt.plot(
    range(n_train, len(y_test) + n_train), y_test, '-', label='test')
  plt.plot(
    range(n_train), y_pred_train, '--', label='prediction train')
  plt.plot(
    range(n_train, len(y_pred) + n_train), y_pred, '--', label='prediction test')
  plt.legend(loc=(1.01, 0))
  plt.xlabel('Date')
  plt.ylabel('Rentals')
  plt.show()


regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# eval_on_features(X, y, regressor)
# The random forest fails to make accurate predictions
# the R^2 score is -0.04, no learning occurred
# The POSIX time is not a useful feature


X_hour = citibike.index.hour.values.reshape(-1, 1)
# eval_on_features(X_hour, y, regressor)
# Using the hour of the day as a feature yielded better
# results, an R^2 score of 0.60


X_hour_week = np.hstack([
  citibike.index.dayofweek.values.reshape(-1, 1),
  citibike.index.hour.values.reshape(-1, 1),
])
# eval_on_features(X_hour_week, y, regressor)
# Adding the day of the week also helped improve the accuracy
# this time we got an R^2 score of 0.84


# eval_on_features(X_hour_week, y, LinearRegression())
# Using a linear model yielded an R^2 score of 0.13, not very good
# this is because linear models assume features are continuous,
# not categorical


enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
# eval_on_features(X_hour_week_onehot, y, Ridge())
# This gives us an R^2 score of 0.62, better than treating
# the features as continuous variables


poly_transformer = PolynomialFeatures(
  degree=2, interaction_only=True, include_bias=False)
X_hour_week_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_poly, y, lr)
# Here we get an R^2 score of 0.85 by adding polynomial features
# this performs just as well as the random forest


hour = ['%02d:00' % i for i in range(0, 24, 3)]
day = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
features = day + hour
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel('Feature name')
plt.ylabel('Feature amplitude')
plt.show()
# Plots the coefficients of each feature and interaction
