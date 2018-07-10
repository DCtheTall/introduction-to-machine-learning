"""
Preprocessing by Scaling
------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


# mglearn.plots.plot_scaling()
# plt.show()
# Plots different types of scaling you can do on a dataset


cancer = load_breast_cancer()
X_train, X_test, _, _ = train_test_split(
  cancer.data, cancer.target, random_state=1)
# print X_train.shape
# print X_test.shape
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
# print 'Transformed shape:\n{}'.format(X_train_scaled.shape)
# print 'Min before scaling:\n{}'.format(X_train.min(axis=0))
# print 'Max before scaling:\n{}'.format(X_train.max(axis=0))
# print 'Min after scaling:\n{}'.format(X_train_scaled.min(axis=0)) # All 0's
# print 'Max after scaling:\n{}'.format(X_train_scaled.max(axis=0)) # All 1's
# Printing the min and maxes of the data by feature before and after a MinMaxScaler transforms the dataset


X_test_scaled = scaler.transform(X_test)
# print 'Transformed shape:\n{}'.format(X_test_scaled.shape)
# print 'Min before scaling:\n{}'.format(X_test.min(axis=0))
# print 'Max before scaling:\n{}'.format(X_test.max(axis=0))
# print 'Min after scaling:\n{}'.format(X_test_scaled.min(axis=0))
# print 'Max after scaling:\n{}'.format(X_test_scaled.max(axis=0))
# Printing the min and max test data by feature before and after the same transformation
# The ranges after the scale can be outside [0, 1] since the scaler is fit to the training data


X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
# Original data
# axes[0].scatter(
#   X_train[:,0], X_train[:,1], c=mglearn.cm2(0), label="Training set", s=60)
# axes[0].scatter(
#   X_test[:,0], X_test[:,1], c=mglearn.cm2(1), marker="^", label="Test set", s=60)
# axes[0].legend('upper left')
# axes[0].set_title('Original data')
# # Properly scaled data
# scaler = MinMaxScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# axes[1].scatter(
#   X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
# axes[1].scatter(
#     X_test_scaled[:, 0], X_test_scaled[:, 1], c=mglearn.cm2(1), marker="^", label="Test set", s=60)
# axes[1].set_title('Properly scaled data')
# # Improperly scaled data - DO NOT DO THIS
# test_scaler = MinMaxScaler().fit(X_test)
# X_test_scaled_badly = test_scaler.transform(X_test)
# axes[2].scatter(
#   X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
# axes[2].scatter(
#   X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], c=mglearn.cm2(1), marker="^", label="Test set", s=60)
# axes[2].set_title('Improperly scaled data')
# plt.show()
# Plotting artificial data before scaling, after proper scaling, and what happens when you
# improperly scale the test data by fitting the scaler to the test data after already scaling
# the training data


X_train, X_test, y_train, y_test = train_test_split(
  cancer.data, cancer.target, random_state=0)
svm = SVC(C=100).fit(X_train, y_train)
# print 'Test set accuracy: {:.2f}'.format(svm.score(X_test, y_test)) # .63
# Now let's try scaling the data first
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm = SVC(C=100).fit(X_train_scaled, y_train)
# print 'Test set accuracy: {:.2f}'.format(svm.score(X_test_scaled, y_test)) # .97 much better
# Trying another scaling algorithm
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm = SVC(C=100).fit(X_train_scaled, y_train)
print 'Test set accuracy: {:.2f}'.format(svm.score(X_test_scaled, y_test)) # .96 not much different
