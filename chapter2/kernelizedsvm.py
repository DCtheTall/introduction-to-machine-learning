"""
Kernelized Support Vector Machines
----------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.svm import LinearSVC, SVC
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.model_selection import train_test_split


X, y = make_blobs(centers=4, random_state=8)
y = y % 2
model = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_separator(model, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
# Plots dataset of 4 clusters of points that
# belong to 2 classes that linear classification
# does not do a good job generalizing for


X_new = np.hstack([X, X[:, 1:] ** 2])
mask = y == 0
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
ax.scatter(
  X_new[mask, 0],
  X_new[mask, 1],
  X_new[mask, 2],
  c='b',
  cmap=mglearn.cm2,
  s=60,
  edgecolor='k',
)
ax.scatter(
  X_new[~mask, 0],
  X_new[~mask, 1],
  X_new[~mask, 2],
  c='r',
  marker='^',
  cmap=mglearn.cm2,
  s=60,
  edgecolor='k',
)
ax.set_xlabel('feature0')
ax.set_ylabel('feature1')
ax.set_zlabel('feature1 ** 2')
plt.show()
# 3D plot of the scatter plot from before
# where the z-axis is the 2nd feature squared
# this plot allows us to make a linear
# classification


model = LinearSVC().fit(X_new, y)
coef, intercept = model.coef_.ravel(), model.intercept_
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 0].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax = Axes3D(figure, elev=-152, azim=-26)
figure = plt.figure()
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(
  X_new[mask, 0],
  X_new[mask, 1],
  X_new[mask, 2],
  c='b',
  cmap=mglearn.cm2,
  s=60,
  edgecolor='k',
)
ax.scatter(
  X_new[~mask, 0],
  X_new[~mask, 1],
  X_new[~mask, 2],
  c='r',
  marker='^',
  cmap=mglearn.cm2,
  s=60,
  edgecolor='k',
)
ax.set_xlabel('feature0')
ax.set_ylabel('feature1')
ax.set_zlabel('feature1 ** 2')
plt.show()
# This is a 3D plot of the decision boundary
# of the new model which takes into consideration
# non-linear terms


ZZ = YY ** 2
dec = model.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contour(
  XX,
  YY,
  dec.reshape(XX.shape),
  levels=[dec.min(), 0, dec.max()],
  cmap=mglearn.cm2,
  alpha=0.5,
)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
# Plots the decision boundary of the model
# which includes feature1 ** 2 as a third feature


X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.show()
# Plots decision boundary for an SVM on the forge dataset
# support vectors are plotted larger


fig, axes = plt.subplots(3, 3, figsize=(3, 3))
for ax, C in zip(axes, [-1, 0, 1]):
  for a, gamma in zip(ax, range(-1, 2)):
    mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(['class 0', 'class 1', 'sv class 0', 'sv class 1'], ncol=4, loc=(.9, 1.2))
plt.show()
# Plots a range of gamma and C (regularization parameter) values
# to see their effect on the model
# High values of gamma means fewer points can influence the decision boundary
# High values of the regularization parameter C mean less regularization


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
  cancer.data, cancer.target, random_state=0
)
svc = SVC().fit(X_train, y_train)
print 'Training set accuracy: {:.3f}'.format(svc.score(X_train, y_train)) # 1.
print 'Test set accuracy: {:.3f}'.format(svc.score(X_test, y_test)) # 0.629
# This model is overfitting substantially, this is due to the fact
# that SVMs work best when the features are of relatively equal magnitude


plt.boxplot(X_train, manage_xticks=False)
plt.yscale('symlog')
plt.xlabel('Feature index')
plt.ylabel('Feature magnitnude')
plt.show()
# Plot showing feature magnitudes of cancer data set are not relatively equal


min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
print 'Minimum for each feature\n{}'.format(X_train_scaled.min(axis=0))
print 'Maximum for each feature\n{}'.format(X_train_scaled.max(axis=0))
# Scale the features to be in the range [0, 1]

X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC().fit(X_train_scaled, y_train)
print 'Training set accuracy: {:.3f}'.format(svc.score(X_train_scaled, y_train)) # .948
print 'Test set accuracy: {:.3f}'.format(svc.score(X_test_scaled, y_test)) # .951
# This did much better, and is actually underfitting the data now


svc = SVC(C=1e3).fit(X_train_scaled, y_train)
print 'Training set accuracy: {:.3f}'.format(svc.score(X_train_scaled, y_train)) # .988
print 'Test set accuracy: {:.3f}'.format(svc.score(X_test_scaled, y_test)) # .972
# Adding complexity (lowering regularization by tuning C) to the model helped improve the accuracy
