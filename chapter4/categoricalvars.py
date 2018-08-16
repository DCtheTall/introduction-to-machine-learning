"""
Categorical variables
---------------------

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


adult_path = os.path.join(mglearn.datasets.DATA_PATH, 'adult.data')
data = pd.read_csv(
  adult_path,
  header=None,
  index_col=False,
  names=[
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
  ],
)
data = data[[
  'age',
  'workclass',
  'education',
  'gender',
  'hours-per-week',
  'occupation',
  'income',
]]
# display(data.head())
# Display data from the mglearn adult dataset


# print(data.gender.value_counts())
# Use .value_counts to make sure that categorical variables are useable,
# i.e. there aren't many unuseable values due to bad data input


# print 'Original featurs:\n', list(data.columns), '\n'
data_dummies = pd.get_dummies(data)
# print 'Features after get_dummies:\n', list(data_dummies.columns)
# pandas has a get_dummies method which will extract dummy variables
# from each categorical variable


# display(data_dummies.head())
# Printing the new DataFrame with dummy variables
# NOTE: do not separate the output variable into dummy variables!


features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
# print 'X.shape: {}  y.shape: {}'.format(X.shape, y.shape)
# So let's extract the output feature and construct a useable dataset
# for a supervised model
# NOTE: pandas DataFrame slices include the last index supplied


X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg = LogisticRegression().fit(X_train, y_train)
# print 'Test score: {:.2f}'.format(logreg.score(X_test, y_test))
# NOTE it is important to create dummy variables before train_test_split


demo_df = pd.DataFrame({
  'Integer Feature': [0, 1, 2, 1],
  'Categorical Feature': ['socks', 'fox', 'socks', 'box'],
})
# display(demo_df)
# display(pd.get_dummies(demo_df))
# Pandas by default only makes dummy variables for features
# with string values, not integers


demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
display(pd.get_dummies(
  demo_df, columns=['Integer Feature', 'Categorical Feature']))
# One way around this (since its not uncommon for categorical variables to be ints)
# is to type cast them to strings
