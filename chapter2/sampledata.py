"""
Sample data examples

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer, load_boston


X, y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(['Class 0', 'Class 1'], loc=4)
# plt.xlabel('First feature')
# plt.ylabel('Second feature')
# plt.show()
# print 'X.shape: {}'.format(X.shape)


X, y = mglearn.datasets.make_wave(n_samples=100)
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel('Feature')
# plt.ylabel('Target')
# plt.show()


cancer = load_breast_cancer()
# print 'Cancer keys():\n{}'.format(cancer.keys())
# print 'Shape of cancer data:\n{}'.format(cancer['data'].shape)
# print 'Sample counts per class:\n{}'.format(
#   {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
# )
# print 'Feature names:\n{}'.format(cancer['feature_names'])


boston = load_boston()
# print 'Data shape:\n{}'.format(boston['data'].shape)
X, y = mglearn.datasets.load_extended_boston()
print 'X.shape: {}'.format(X.shape)
