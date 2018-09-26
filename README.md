<a href="https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413"><img alt="Introduction to Machine Learning with Python Cover" src="./cover.png" style="width: 200px; height: auto; padding: 10px;"></a>

Code examples from:
# Introduction to Machine Learning with Python
[Buy here](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)

## Table of Contents

1. [Chapter 1](#chapter-1)&nbsp; Introduction
2. [Chapter 2](#chapter-2)&nbsp; Supervised Learning
3. [Chapter 3](#chapter-3)&nbsp; Unsupervised Learning
4. [Chapter 4](#chapter-4)&nbsp; Representing Data and Engineering Features
5. [Chapter 5](#chapter-5)&nbsp; Model Evaluation and Improvement
6. [Chapter 6](#chapter-6)&nbsp; Algorithm Chains and Pipelines
7. [Chapter 7](#chapter-7)&nbsp; Working with Text Data
8. [Chapter 8](#chapter-8)&nbsp; Wrapping Up

---

## Chapter 1
### Introduction

The code examples from the first chapter are split into `irises.py` and `libs.py`.

The latter, `libs.py`, just has some examples of methods and instances of classes
from the libraries the rest of the code will use (like numpy). `irises.py` contains
and implementation of the first model the book covers, the k-Nearest Neighbor
classifier.

---

## Chapter 2
### Supervised Learning

This chapter covers multiple supervised learning algorithms. Supervised learning
is done when you fit a predictive model with training data where the outcome
is already measured. These models are then evaluated by their ability to generalize,
or accurately predict test data it has not seen during the fitting process.

The `.py` files are broken down by model type (for the most part). Below
is a table of content with links to each algorithm covered by the code.

1. [k-Nearest Neighbors](#k-nearest-neighbors)
2. [Linear Regression](#linear-regression)
3. [Ridge Regression](#ridge-regression)
4. [Lasso Regression](#lasso)
4. [LogisticRegression](#logistic-regression)
5. [Naive Bayes Classifiers](#naive-bayes-classifiers)
6. [Decision Trees](#decision-trees)
7. [Kernelized Support Vector Machines](#kernelized-support-vector-machines)
8. [Neural Networks](#neural-networks)
9. [Predicting Uncertainty](#predicting-uncertainty)

#### k-Nearest Neighbors

k-Nearest Neighbors is the simplest supervised classifiers
covered in this repository. It classifies new data by finding the
k closest known data points to the new one and classifying the
new data as the class of the majority.

The k-Nearest Neighbors Regressor is a regression algorithm which
linearly interpolates the output of a new set of input features
from the k closest known data points.

The parameters covered in this repo are:

- `n_neighbors` The number of neighbors taken into consideration.

#### Linear Regression

Linear regression is a well-known algorithm which tries to find
a best fit linear relationship between known data's features and
its ouput.

#### Ridge Regression

Ridge regression is a form of linear regression which dampens
the impact of features whose coefficients in linear regression
are far from 0.

The parameters covered in this repo are:

- `alpha` The regularization constant, a higher `alpha` means more regularization, when `alpha` is 0, ridge regression becomes normal linear regression.

#### Lasso Regression

Lasso regression is also linear regression with regularization. The difference
is that lasso regression uses the sum of each feature vector's components
whereas linear regression uses the sum of the components' squares.

The parameters covered in this repo are:

- `alpha` The regularization constant, a higher `alpha` means more regularization, when `alpha` is 0, lasso regression becomes normal linear regression.

#### Logistic Regression

Logistic regression is a classifier algorithm which uses regularized linear
regression. For two-class decisions, it uses the sign of the output of linear
regression to make its decisions. For multi-class, it compares each individual
class to every other one as if it were making a two-class decision, then picks
the class with the best fit.

The parameters covered in this repo are:

- `C` The regularization constant, as `C` increases the regularization decreases
- `penalty` What type of penalty to use for regularization. The default is to use
the Euclidean metric (ridge regression). The code also has an example of using the
L1 metric (lasso regression).

#### Naive Bayes Classifiers

This repo has 3 examples of naive bayes classifiers:

1. **BernoulliNB:** This classifier works on data where the feature set is a vector with binary values (0 or 1).

2. **MultinomialNB:** This classifier works on data where the feature set is a vector of discrete values (integers).

3. **GaussianNB:** This classifier works on data where the feature set is a real-valued vector.

#### Decision Trees

Decision trees are a type of classifier which classifies data using a series
of conditionals. It can be fit to training data perfectly, but in this case
is unable to generalize.

One way to improve the generalization performance is to set a max depth for
the tree. This is called _pruning_ the tree.

Decision trees can be used for regression as well and can fit training data
perfectly at the expense of losing the ability to generalize.

Parameters covered in this repo are:

- `max_depth` which controls the maximum depth of the decision tree

Another way to increase the generalization performance is to use multiple
decision trees in one classifier. One way to do this is with the
`RandomForestClassifier` which creates a number of decision trees
and then classifies new data probabilistically.

Parameters for `RandomForestClassifier` covered in this repo are:

- `n_estimators` which controls the number of decision trees trained in the ensemble.

Gradient boosted decision trees are an ensemble which learn from each previous
tree. These trees can generalize very well without any parameter tuning.

The parameters covered in this repo are:

- `max_depth` which controls the maximum depth of the decision tree
- `learning_rate` which controls how much influence each tree has, the smaller
the `learning_rate`, the less influence each previous iteration has on the ensemble.

#### Kernelized Support Vector Machines

Kernelized SVM's are a supervised learning algorithm which use linear
regression as well as non-linear terms to generalize to new data.

The parameters covered are:

- `gamma` Higher values of `gamma` mean less points can influence the decision
- `C` Higher values of `C` mean there will be less regularization

#### Neural Networks

Neural networks are a famous classifying algorithm which work by
doing multiple layers of linear regression then applying a non-linear
activation function on each layer so that each level of regression
can introduce more non-linear terms to the decision boundary. Common
activation functions are `atan(x)` and `relu(x) = max(x, 0)`.

Neural networks allow for very non-linear decision boundaries with
very high accuracy. Though they are computationally expensive to train.

The parameters covered for neural networks are:

- `hidden_layers` a list of integers which determines how many hidden layers and how many hidden nodes the neural network will use.
- `alpha` a regularization constant for each step of linear regression

#### Predicting Uncertainty

One way to measure the uncertainty of a classifier is
to use the `.predict_proba` method which shows how certain the classifier
is with its decision when classifying new data.

The `.decision_function` method takes input data and returns the
predictions for each point in the input.

---

TODO Chapters 3-8
