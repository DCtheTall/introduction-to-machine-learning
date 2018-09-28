"""
k-Means Clustering
------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import make_blobs, make_moons, fetch_lfw_people
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA


mglearn.plots.plot_kmeans_algorithm()
plt.show()


mglearn.plots.plot_kmeans_boundaries()
plt.show()


X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print 'Cluster memberships:\n{}'.format(kmeans.labels_)
print kmeans.predict(X)
# kmeans takes an arbitrary dataset and kinds
# k classes to classify points by distance


mglearn.discrete_scatter(X[:,0], X[:,1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
  kmeans.cluster_centers_[:, 0],
  kmeans.cluster_centers_[:, 1],
  [0, 1, 2],
  markers='^', markeredgewidth=2)
plt.show()
# kMeans also stores the coordinates of each cluster center


fig, axes = plt.subplots(1, 2, figsize=(10, 5))
kmeans = KMeans(n_clusters=2).fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])
kmeans = KMeans(n_clusters=5).fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
plt.show()
# Plot kMeans on the same dataset at k = 2, 5


X_varied, y_varied = make_blobs(
  n_samples=100, cluster_std=[1., 2.5, .5], random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
mglearn.discrete_scatter(X_varied[:,0], X_varied[:,1], y_pred)
plt.legend(['cluster ' + str(i) for i in range(3)], loc='best')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
# This data has 2 clusters that are well defined on the right
# and then a sparse group of points in the middle. This causes
# some sparse points near the well defined clusters to be
# erroneously classified


X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)
kmeans = KMeans(n_clusters=3).fit(X)
y_pred = kmeans.predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
  kmeans.cluster_centers_[:, 0],
  kmeans.cluster_centers_[:, 1],
  [0, 1, 2],
  markers='^', markeredgewidth=2)
plt.show()
# kMeans also fails to properly classify the points in this dataset
# because two of the clusters are very close and the data is spread
# about a diagonal


X, y = make_moons(n_samples=200, noise=.05, random_state=0)
kmeans = KMeans(n_clusters=2).fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolor='k')
plt.scatter(
  kmeans.cluster_centers_[:,0],
  kmeans.cluster_centers_[:,1],
  marker='^',
  c=[mglearn.cm2(0), mglearn.cm2(1)],
  s=100,
  edgecolors='k')
plt.show()
# Here we see kMeans also fails for clusters
# with complex shapes


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1
image_shape = people.images[0].shape
X_people = people.data[mask]
X_people = X_people / 255.
y_people = people.target[mask]
X_train, X_test, y_train, y_test = train_test_split(
  X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=100, random_state=0).fit(X_train)
pca = PCA(n_components=100, random_state=0).fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0).fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

fig, axes = plt.subplots(3, 5,
  subplot_kw={'xticks': (), 'yticks': ()}, figsize=(8, 8))
fig.suptitle('Extracted Components')
for ax, comp_kmeans, comp_pca, comp_nmf in zip(
  axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
  ax[0].imshow(comp_kmeans.reshape(image_shape))
  ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
  ax[2].imshow(comp_nmf.reshape(image_shape))
axes[0, 0].set_ylabel('kmeans')
axes[1, 0].set_ylabel('pca')
axes[2, 0].set_ylabel('nmf')
plt.show()
# Plot of 100 components after deconstruction by PCA, NMF, and kMeans

fig, axes = plt.subplots(4, 5,
  subplot_kw={'xticks': (), 'yticks': ()}, figsize=(8, 8))
fig.suptitle('Reconstructions')
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
  axes.T,
  X_test,
  X_reconstructed_kmeans,
  X_reconstructed_pca,
  X_reconstructed_nmf,
):
  ax[0].imshow(orig.reshape(image_shape))
  ax[1].imshow(rec_kmeans.reshape(image_shape))
  ax[2].imshow(rec_pca.reshape(image_shape))
  ax[3].imshow(rec_nmf.reshape(image_shape))
axes[0, 0].set_ylabel('original')
axes[1, 0].set_ylabel('kmeans')
axes[2, 0].set_ylabel('pca')
axes[3, 0].set_ylabel('nmf')
plt.show()
# Plot reconstruction of the faces using kMeans, PCA, and NMF
# for kMeans the algorithm will choose the mean closest to
# the face


X, y = make_moons(n_samples=200, noise=.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred, s=60, cmap='Paired')
plt.scatter(
  kmeans.cluster_centers_[:,0],
  kmeans.cluster_centers_[:,1],
  s=60, marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
print 'Cluster memberships:\n{}'.format(y_pred)
plt.show()
# kMeans can provide insight on complex data with a high number of means calculated


distance_features = kmeans.transform(X)
print 'Distance feature shape: {}'.format(distance_features.shape)
print 'Distance features:\n{}'.format(distance_features)
# Prints the distance of each element in the set from the means
