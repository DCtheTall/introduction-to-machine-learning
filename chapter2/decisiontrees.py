"""
Decision Trees
--------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
import graphviz
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


mglearn.plots.plot_animal_tree()
plt.show()
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
print 'Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train))
print 'Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test))
# Training: 100%
# Test: 93.7%
# This model is overfitting because the all leafs of the tree
# are pure, and it can go to an arbitrary depth


tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
print 'Accuracy on training set: {:.3f}'.format(tree.score(X_train, y_train)) # .988
print 'Accuracy on test set: {:.3f}'.format(tree.score(X_test, y_test)) # .951
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
with open('tree.dot') as f:
  dot_graph = f.read()
display(graphviz.Source(dot_graph))
# Generates a .dot file that graphviz can turn into an
# image displaying the decision tree model


print 'Feature importances:\n{}'.format(tree.feature_importances_)
def plot_feature_importances(model):
  n_features = cancer.data.shape[1]
  plt.barh(range(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), cancer.feature_names)
  plt.xlabel('Feature importance')
  plt.ylabel('Feature')
  plt.ylim(-1, n_features)
plot_feature_importances(tree)
plt.show()
# Plots the feature importances (in [0, 1], sum is 1) as a bar graph
# Closer to 1 means the


tree = mglearn.plots.plot_tree_not_monotone()
plt.show()
display(tree)


ram_prices = pd.read_csv(
  os.path.join(
    mglearn.datasets.DATA_PATH,
    'ram_price.csv',
  ),
)
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('Year')
plt.ylabel('Price in $/Mbyte')
plt.show()
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


X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2).fit(X_train, y_train)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
  ax.set_title('Tree {}'.format(i))
  mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1,-1], alpha=.4)
axes[-1,-1].set_title('Random Forest')
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
plt.show()
# Plots each decision tree in the random forest classifier then plots the decision
# boundaries made by the ensemble of randomly constructed decision trees


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print 'Training set accuracy: {:.3f}'.format(forest.score(X_train, y_train)) # 1.00
print 'Test set accuracy: {:.3f}'.format(forest.score(X_test, y_test)) # 0.972
# Scoring an ensemble of 100 randomized decision trees on the breast cancer data set
# This model outperforms the linear models without any tuning of parameters


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print 'Training set score: {:.3f}'.format(model.score(X_train, y_train))
print 'Test set score: {:.3f}'.format(model.score(X_test, y_test))
# Training set score: 1.000
# Test set score: 0.958
# Default params are max-depth: 3 and learning rate: 0.1
# This model is overfitting, let's try making it less complex


model = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train, y_train)
print 'Training set score: {:.3f}'.format(model.score(X_train, y_train))
print 'Test set score: {:.3f}'.format(model.score(X_test, y_test))
# Training set score: 0.991
# Test set score: 0.972
# Reducing the maximum depth of the gradient boosted ensemble improved generalization
# This is expected, considering previous settings overfit


model = GradientBoostingClassifier(random_state=0, learning_rate=0.01).fit(X_train, y_train)
print 'Training set score: {:.3f}'.format(model.score(X_train, y_train))
print 'Test set score: {:.3f}'.format(model.score(X_test, y_test))
# Training set score: 0.988
# Test set score: 0.965
# Reducing the learning rate of the gradient boosted ensemble also helped


model = GradientBoostingClassifier(random_state=0, max_depth=1).fit(X_train, y_train)
plot_feature_importances(model)
plt.show()
# Plotting feature importance for the gradient boosting classifier
# One can see it may choose to completely ignore some features
