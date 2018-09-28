"""
DBSCAN
------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler


X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print 'Cluster memberships:\n{}'.format(clusters)
# Running DBSCAN on this small dataset without modifying min_samples or eps
# results in it labeling all points as noise


mglearn.plots.plot_dbscan()
plt.show()
# Plots examples of DBSCAN
# white points are noise
# Core samples are the larger marks
# Boundaries are smaller


X, y = make_moons(n_samples=200, noise=.05, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
plt.scatter(
  X_scaled[:,0], X_scaled[:,1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
# Plots how DBSCAN with default settings
# perfectly classifies the two moons dataset
