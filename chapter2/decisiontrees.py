"""
Decision Trees
--------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
import graphviz
import os
from sklearn.linear_model import LinearRegression


# mglearn.plots.plot_animal_tree()
# plt.show()
# Plot visual of decision tree which classifies animals
# with verbal questions for the point of demonstration


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
  cancer.data,
  cancer.target,
  stratify=cancer.target,
  random_state=42,
)
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
# print 'Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train))
# print 'Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test))
# Training: 100%
# Test: 93.7%
# This model is overfitting because the all leafs of the tree
# are pure, and it can go to an arbitrary depth


tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
# print 'Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train))
# print 'Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test))
# Add a maximum depth of 4 to pre-prune the tree, which gives better
# generalization results


export_graphviz(
  tree,
  out_file='tree.dot',
  class_names=['malignant', 'benign'],
  feature_names=cancer.feature_names,
  impurity=False,
  filled=True,
)
# with open('tree.dot') as f:
  # dot_graph = f.read()
# display(graphviz.Source(dot_graph))
# Generates a .dot file that graphviz can turn into an
# image displaying the decision tree model


# print 'Feature importances:\n{}'.format(tree.feature_importances_)
def plot_feature_importances(model):
  n_features = cancer.data.shape[1]
  plt.barh(range(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), cancer.feature_names)
  plt.xlabel('Feature importance')
  plt.ylabel('Feature')
  plt.ylim(-1, n_features)
# plot_feature_importances(tree)
# plt.show()
# Plots the feature importances (in [0, 1], sum is 1) as a bar graph
# Closer to 1 means the


# tree = mglearn.plots.plot_tree_not_monotone()
# plt.show()
# display(tree)


ram_prices = pd.read_csv(
  os.path.join(
    mglearn.datasets.DATA_PATH,
    'ram_price.csv',
  ),
)
# plt.semilogy(ram_prices.date, ram_prices.price)
# plt.xlabel('Year')
# plt.ylabel('Price in $/Mbyte')
# plt.show()
# Plot of RAM prices over time using a logarithmic scale
# for the y-axis


data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]
X_train = data_train.date[:,np.newaxis]
y_train = np.log(data_train.price)
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)
plt.semilogy(data_train.date, data_train.price, label='Training data')
plt.semilogy(data_test.date, data_test.price, label='Test data')
plt.semilogy(ram_prices.date, price_tree, label='Tree prediction')
plt.semilogy(ram_prices.date, price_lr, label='Linear prediction')
plt.legend()
plt.show()
# Plots the ram prices over time alongside the predictions from
# a linear model and a deicision tree regression

# The linear model finds the best fit linear curve to the test
# data which provides a reasonable prediction for the training
# and test data

# The decision tree regression predicts the training data perfectly
# but cannot make any predictions outside the range of the training
# data
