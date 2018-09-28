"""
Agglomerative Clustering
------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, ward


mglearn.plots.plot_agglomerative_algorithm()
plt.show()
# Plot diagram of agglomerative clustering in action


X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(['Cluster 0', 'Cluster 1', 'Custer 2'], loc='best')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
# Agglomerative clustering perfectly classifies
# 3 well-defined blobs of data points


mglearn.plots.plot_agglomerative()
plt.show()
# Shows step by step how the clustering algorithm
# combines clusters


X, y = make_blobs(random_state=0, n_samples=12)
# Apply the ward clustering to the dataset X
linkage_arr = ward(X)
# Now we plot the dendogram for the linkage_arr containing the
# distances
dendrogram(linkage_arr)
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel('Sample index')
plt.ylabel('Cluster distance')
plt.show()
# Plots dendogram of the clustering which draws horizontal lines where
# the split into 3 and 2 clusters
