"""
Comparing Clustering Algorithms
-------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import make_moons, fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, ward


X, y = make_moons(n_samples=200, noise=.05, random_state=0)
X_scaled = StandardScaler().fit_transform(X)
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
algos = [
  KMeans(n_clusters=2),
  AgglomerativeClustering(n_clusters=2),
  DBSCAN(),
]
fig, axes = plt.subplots(
  1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})


# Adjusted random index (ARI)
axes[0].scatter(
  X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
axes[0].set_title(
  'Random Assignment: {:.2f}'.format(
    adjusted_rand_score(y, random_clusters)))
for ax, algo in zip(axes[1:], algos):
  clusters = algo.fit_predict(X_scaled)
  ax.scatter(
    X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
  ax.set_title(
    '{}: {:.2f}'.format(
        algo.__class__.__name__, adjusted_rand_score(y, clusters)))
plt.show()
# ARI scores unrelated clusters 0 and related clusters 1
# Random assignment scores a 0 and DBSCAN scores a perfect 1


# Silhouette score
# plot random assignment
axes[0].scatter(
  X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
axes[0].set_title(
  'Random Assignment: {:.2f}'.format(
    silhouette_score(X_scaled, random_clusters)))
# plot algorithms
for ax, algo in zip(axes[1:], algos):
  clusters = algo.fit_predict(X_scaled)
  ax.scatter(
    X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
  ax.set_title(
    '{}: {:.2f}'.format(
      algo.__class__.__name__, silhouette_score(X_scaled, clusters)))
plt.show()
# Silhouette score measures how close together the clusters are
# Here kMeans gets the best silhouette score even tho DBSCAN
# is a preferable clustering algo


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1
image_shape = people.images[0].shape
X_people = people.data[mask]
X_people = X_people / 255.
y_people = people.target[mask]


# use PCA to extract eignenfaces from lfw data
uniq_labels = lambda labels: 'Unique labels: {}'.format(np.unique(labels))
pca = PCA(n_components=100, whiten=True, random_state=0)
X_pca = pca.fit_transform(X_people)


# apply DBSCAN with default parameters
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print uniq_labels(labels)
# first pass labels all points as noise, lets try lowering min_samples
dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print uniq_labels(labels)
# Still everything is noise, so lets increase eps
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print uniq_labels(labels)


print 'Num points per cluster: {}'.format(np.bincount(labels + 1))
# 35 faces are considered noise
# 2028 faces are in one cluster


noise = X_people[labels == -1]
fig, axes = plt.subplots(
  5, 7, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
for img, ax in zip(noise, axes.ravel()):
  ax.imshow(img.reshape(image_shape), vmin=0, vmax=1)
plt.show()
# Mostly faces looking at an angle or cropped, some people
# wearing hats, or holding something in front of their face


for eps in range(1, 14, 2): # odd numbers in [1, 13]
  dbscan = DBSCAN(min_samples=3, eps=eps)
  labels = dbscan.fit_predict(X_pca)
  print 'eps = {}'.format(eps)
  print 'Number of clusters: {}'.format(len(np.unique(labels)))
  print 'Cluster sizes: {}'.format(np.bincount(labels + 1))
# Prints the number of clusters and cluster sizes for different
# eps settings. Most faces are separated into one larger class
# and noise, with some small clusters which are probably some
# very distinct faces


dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels) + 1):
  mask = labels == cluster
  n_images = np.sum(mask)
  fig, axes = plt.subplots(
    1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})
  for img, label, ax in zip(X_people[mask], y_people[mask], axes):
    ax.imshow(img.reshape(image_shape), vmin=0, vmax=1)
    ax.set_title(people.target_names[label].split()[-1])
plt.show()
# Plotting the clusters of faces determined by DBSCAN
# some clusters correspond to a particular person


km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print 'Cluster sizes k means: {}'.format(np.bincount(labels_km))
# Divides it into 10 relatively similarly sized clusters


fig, axes = plt.subplots(
  2, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
for center, ax in zip(km.cluster_centers_, axes.ravel()):
  ax.imshow(
    pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
plt.show()
# Each cluster center is a smooth face, which makes sense given its the
# average of on the order of 10^2 faces


mglearn.plots.plot_kmeans_faces(
  km, pca, X_pca, X_people, y_people, people.target_names)
plt.show()
# plots each center, the 5 closest faces to center,
# and the 5 farthest in each cluster
# As expected faces closer to the smoothed faces are facing
# similar directions and have similar facial expressions
# Faces that are far from center may have different orientations,
# headwear, or facial expressions


agglom = AgglomerativeClustering(n_clusters=10)
labels_agg = agglom.fit_predict(X_pca)
print 'Cluster sizes: {}'.format(np.bincount(labels_agg))
# Like kMeans, it creates relatively similarly sized clusters
# print 'ARI: {:.2f}'.format(adjusted_rand_score(labels_agg, labels_km))
# They seem to be rather uncorrelated (0.09)

linkage_arr = ward(X_pca)
plt.figure(figsize=(20, 5))
dendrogram(
  linkage_arr, p=7, truncate_mode='level', no_labels=True)
plt.xlabel('Sample index')
plt.ylabel('Cluster distance')
plt.show()
# The plot shows branches vary in length
# There does not seem to be a good cutoff for
# classifying the data


for cluster in range(10):
  mask = labels_agg == cluster
  fig, axes = plt.subplots(
    1, 10, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
  axes[0].set_ylabel(np.sum(mask))
  for img, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
    ax.imshow(
      img.reshape(image_shape), vmin=0, vmax=1)
    ax.set_title(
      people.target_names[label].split()[-1], fontdict={'fontsize': 9})
plt.show()


agglom = AgglomerativeClustering(n_clusters=40)
labels_agg = agglom.fit_predict(X_pca)
print 'Cluster sizes: {}'.format(np.bincount(labels_agg))
# Clusters now vary in size, some are only a few pics whereas
# some are a few thousand


for cluster in [10, 13, 19, 22, 36]: # hand picked clusters
  mask = labels_agg == cluster
  fig, axes = plt.subplots(
    1, 15, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
  cluster_size = np.sum(mask)
  axes[0].set_ylabel(
    '#{}: {}'.format(cluster, cluster_size))
  for img, label, asdf, ax in zip(
    X_people[mask], y_people[mask], labels_agg[mask], axes):
      ax.imshow(
        img.reshape(image_shape), vmin=0, vmax=1)
      ax.set_title(
        people.target_names[label].split()[-1], fontdict={'fontsize': 9})
  for i in range(cluster_size, 15):
    axes[i].set_visible(False)
plt.show()
# Demonstration of classification with agglomerative clustering
