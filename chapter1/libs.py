"""
Chapter 1
---------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import sparse


x = np.array([[1, 2, 3], [4, 5, 6]])
# print 'x:\n{}'.format(x)


eye = np.eye(4)
# print 'Numpy array:\n{}'.format(eye)


"""
On spare matrices: https://en.wikipedia.org/wiki/Sparse_matrix

"""


sparse_matrix = sparse.csr_matrix(eye)
# print '\nSciPy sparse CSR matrix:\n{}'.format(sparse_matrix)


data = np.ones(4)
row_indicies = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indicies, col_indices)))
# print 'COO representation:\n{}'.format(eye_coo)


x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker='x')
# plt.show()


data = {
  'Name': ['John', 'Annay', 'Peter', 'Linda'],
  'Location': ['New York', 'Paris', 'Berlin', 'London'],
  'Age': [24, 13, 53, 33],
}
data_pandas = pd.DataFrame(data)
# display(data_pandas)
