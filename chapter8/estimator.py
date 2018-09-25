"""
Estimator class template
------------------------

"""


from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, first_parameter=1, second_parameter=2):
    # All parameters must be specified in init
    self.first_parameter = 1
    self.second_parameter = 2

  def fit(self, X, y=None):
    # fit should only take X and y as parameters
    # Even if your model is unsupervised, you need to accept a y argument

    # Model fitting code goes here

    return self # Fitting returns self

  def transform(self, X):
    # Apply some transformation to X
    X_transformed = X + 1
    return X_transformed
