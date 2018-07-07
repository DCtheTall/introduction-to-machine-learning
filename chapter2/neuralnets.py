"""
Neural networks
---------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split


# line = np.linspace(-3, 3, 100)
# plt.plot(line, np.tanh(line), label="tanh")
# plt.plot(line, np.maximum(line, 0), label="relu")
# plt.legend(loc="best")
# plt.xlabel("x")
# plt.ylabel("relu(x), tanh(x)")
# plt.show()
# Plots recitifying functions, non-linear functions that
# can be applied to each layer of a neural network, in
# this case relu(x) = max(0, x) and tanh(x)


def plot_decision_boundary(X_train, y_train, model):
  """
  Plots the decision boundary of a classification
  model on a dataset with 2 features

  """
  mglearn.plots.plot_2d_separator(model, X_train, fill=True, alpha=.3)
  mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
  plt.xlabel('Feature 0')
  plt.ylabel('Feature 1')
  plt.show()


X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# mlp = MLPClassifier(solver="lbfgs", random_state=0).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')
# plt.show()
# Plot of a non-linear decision boundary makde by a neural
# network with 100 hidden nodes (default)


# mlp = MLPClassifier(solver="lbfgs", random_state=0, hidden_layer_sizes=[10])
# mlp.fit(X_train, y_train)
# plot_decision_boundary(X_train, y_train, mlp)
# Plot a neural network with only 10 hidden nodes


# mlp = MLPClassifier(solver="lbfgs", random_state=0, hidden_layer_sizes=[10, 10])
# mlp.fit(X_train, y_train)
# plot_decision_boundary(X_train, y_train, mlp)
# Plot a neural network with 10 hidden layers of 10 hidden nodes
# Leads to a more complex model


# mlp = MLPClassifier(
#   solver='lbfgs',
#   activation='tanh',
#   random_state=0,
#   hidden_layer_sizes=[10, 10],
# )
# mlp.fit(X_train, y_train)
# plot_decision_boundary(X_train, y_train, mlp)
# Another, more-nonlinear decision boundary made by setting
# the activation function to tanh(x)


# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# for axx, n_hidden_nodes in zip(axes, [10, 100]):
#   for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
#     mlp = MLPClassifier(
#       solver='lbfgs',
#       random_state=0,
#       hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
#       alpha=alpha,
#     )
#     mlp.fit(X_train, y_train)
#     mglearn.plots.plot_2d_separator(
#       mlp,
#       X_train,
#       fill=True,
#       alpha=.3,
#       ax=ax,
#     )
#     mglearn.discrete_scatter(
#       X_train[:,0],
#       X_train[:,1],
#       y_train,
#       ax=ax,
#     )
#     ax.set_title(
#       'n_hidden=[{}, {}]\nalpha={:.4f}'.format(
#         n_hidden_nodes,
#         n_hidden_nodes,
#         alpha,
#       )
#     )
# plt.show()
# Plotting decision boundaries for neural network as we vary
# the number of hidden nodes and layers and alpha, which
# controls the amount of L2 regularization on each weighted sum
# (lower alpha -> less regularization)


# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# for i, ax in enumerate(axes.ravel()):
#   mlp = MLPClassifier(
#     solver='lbfgs',
#     random_state=i,
#     hidden_layer_sizes=[100, 100],
#   )
#   mlp.fit(X_train, y_train)
#   mglearn.plots.plot_2d_separator(
#     mlp,
#     X_train,
#     fill=True,
#     alpha=.3,
#     ax=ax,
#   )
#   mglearn.discrete_scatter(
#     X_train[:,0],
#     X_train[:,1],
#     y_train,
#     ax=ax,
#   )
# plt.show()
# Neural networks start with a random initialization, these plots
# show how different initial random states can affect models
# learning with smaller datasets


def score_model(X_train, X_test, y_train, y_test, model):
  print 'Accuracy on the training set: {:.3f}'.format(model.score(X_train, y_train))
  print 'Accuracy on the test set: {:.3f}'.format(model.score(X_test, y_test))


cancer = load_breast_cancer()
# print 'Cancer data per-feature maxima:\n{}'.format(cancer.data.max(axis=0))
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
# score_model(X_train, X_test, y_train, y_test, mlp)
# Training score: .91
# Test score: .88
# Data needs to be scaled for neural networks


mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
# mlp.fit(X_train_scaled, y_train)
# score_model(
#   X_train_scaled,
#   X_test_scaled,
#   y_train,
#   y_test,
#   mlp,
# )
# Training score: .991
# Test score: .965
# This result is much better but we get a warning from
# the model that the maximum number of iterations was
# reached, lets increase the maximum iterations


mlp = MLPClassifier(random_state=0, max_iter=1000)
# mlp.fit(X_train_scaled, y_train)
# score_model(
#   X_train_scaled,
#   X_test_scaled,
#   y_train,
#   y_test,
#   mlp,
# )
# Training score: .993
# Test score: .972
# Slightly better performance


mlp = MLPClassifier(max_iter=1000, random_state=0, alpha=1)
mlp.fit(X_train_scaled, y_train)
score_model(
  X_train_scaled,
  X_test_scaled,
  y_train,
  y_test,
  mlp,
)
# Training score: .988
# Test score: .972
# Same performance, it turns out the .972 is because of 4 outliers
# in the dataset


plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input feature')
plt.colorbar()
plt.show()
# Color plot of the weights of each feature in each node of
# the hidden layer
