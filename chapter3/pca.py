"""
Principle Component Analysis
----------------------------

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# mglearn.plots.plot_pca_illustration()
# plt.show()
# Plots an illustration of principal component analysis


cancer = load_breast_cancer()
# fig, axes = plt.subplots(15, 2, figsize=(10, 20))
# malignant = cancer.data[cancer.target == 0]
# begign = cancer.data[cancer.target == 1]
# ax = axes.ravel()
# for i in range(30):
#   _, bins = np.histogram(cancer.data[:,i], bins=50)
#   ax[i].hist(malignant[:,i], bins=bins, color=mglearn.cm3(1), alpha=.5)
#   ax[i].hist(begign[:,i], bins=bins, color=mglearn.cm3(0), alpha=.5)
#   ax[i].set_title(cancer.feature_names[i])
#   ax[i].set_yticks(())
# ax[0].set_xlabel('Feature magnitude')
# ax[0].set_ylabel('Frequency')
# ax[0].legend(['malignant', 'begign'], loc='best')
# fig.tight_layout()
# plt.show()
# Plot of histograms of each feature in the cancer data
# set which features distinguish if a tumor is begign or
# malignant


scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
# print 'Original shape: {}'.format(str(X_scaled.shape))
# print 'Reduced shape: {}'.format(str(X_pca.shape))
# Reduces dataset to a set of 2D vectors


# plt.figure(figsize=(8, 8))
# mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
# plt.legend(cancer.target_names, loc="best")
# plt.gca().set_aspect('equal')
# plt.xlabel('First principal component')
# plt.ylabel('Second principal component')
# plt.show()
# Plot the dataset after principal component analysis
# the data separates quite well as a 2D dataset after
# being processed this way


# print 'PCA component shape: {}'.format(pca.components_.shape)
# List of the importance of each feature
# wrt each of the 2 principal components


# print 'PCA components:\n{}'.format(pca.components_)
# Prints importances of each feature for the 2 components


# plt.matshow(pca.components_, cmap='viridis')
# plt.yticks([0, 1], ['First component', 'Second component'])
# plt.colorbar()
# plt.xticks(
#   range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
# plt.xlabel('Feature')
# plt.ylabel('Principal components')
# plt.show()
# Color plot of each feature importance wrt each of the principal components
# not very easy to interpret


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
# print 'people.images.shape: {}'.format(people.images.shape) # (3023, 87, 65)
# print 'Number of classes: {}'.format(len(people.target_names)) # 62
# Shape of the people data
# Number of different classes


# fig, axes = plt.subplots(
#   2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#   ax.imshow(image)
#   ax.set_title(people.target_names[target])
# plt.show()
# Plot face images


counts = np.bincount(people.target)
# for i, (count, name) in enumerate(zip(counts, people.target_names)):
#   print '{0:25} {1:3}'.format(name, count)
# Prints number of photos of each person
# Some people have significantly more photos than others


mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.
# Limit dataset to only 50 photos per person to avoid skewing


X_train, X_test, y_train, y_test = train_test_split(
  X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
# print 'Test score of 1-nn: {:.2f}'.format(knn.score(X_test, y_test))
# Test score of 23%, not great but much better than random guessing


# mglearn.plots.plot_pca_whitening()
# plt.show()
# Illustration of PCA whitening which scales
# the magnitudes of each PCA so that each component
# has a mean of 0 and a variance of 1


pca = PCA(n_components=100, whiten=True, random_state=0)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# print 'X_train_pca.shape: {}'.format(X_train_pca.shape) # (1547, 100)
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_pca, y_train)
# print 'Test score of 1-nn: {:.2f}'.format(knn.score(X_test_pca, y_test))
# Test accuracy increased to 28%


# print 'pca.components_.shape: {}'.format(pca.components_.shape)
# 100, 5655


# fix, axes = plt.subplots(
#   3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
# for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
#   ax.imshow(component.reshape(people.images[0].shape), cmap='viridis')
#   ax.set_title('{} component'.format(i + 1))
# plt.show()
# Plot image representation of principal components of the image


# mglearn.plots.plot_pca_faces(X_train, X_test, people.images[0].shape)
# plt.show()
# Plot reconstruction of faces using PCA


# mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
# plt.xlabel('First principal component')
# plt.ylabel('Second principal component')
# plt.show()
# 2D scatter plot of 2 principal components
# not much to gather from it, very non-linear
