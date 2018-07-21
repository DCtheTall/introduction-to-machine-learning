"""
Non-negative Matrix Factorization
---------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA


# mglearn.plots.plot_nmf_illustration()
# plt.show()
# Visualization of non-negative matrix factorization
# Only works on non-negative feature values


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.
X_train, X_test, y_train, y_test = train_test_split(
  X_people, y_people, stratify=y_people, random_state=0)
# mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
# plt.show()
# Plot illustration of facial reconstruction using components
# from NMF


nmf = NMF(n_components=15, random_state=0)
X_train_nmf = nmf.fit_transform(X_train)
X_test_nmf = nmf.transform(X_test)
# fix, axes = plt.subplots(
#   3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
# for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
#   ax.imshow(component.reshape(image_shape))
#   ax.set_title('{}. component'.format(i))
# plt.show()
# Plot individual components determined by NMF


def plot_faces_with_large_component(compn):
  """
  Plot all faces with a high value of the i^th component
  provided in the argument
  """
  inds = np.argsort(X_train_nmf[:, compn])[::-1]
  fig, axes = plt.subplots(
      2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
  fig.suptitle('Large component 3')
  for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
  plt.show()


# plot_faces_with_large_component(3)
# Plot faces which have a high value of component 3
# plots male faces looking right


# plot_faces_with_large_component(7)
# Plot faces with a high value of component 7
# plots faces (both sexes) looking left


S = mglearn.datasets.make_signals()
# plt.figure(figsize=(6, 1))
# plt.plot(S, '-')
# plt.xlabel('Time')
# plt.ylabel('Signal')
# plt.show()
# Plots 3 separate signals for 3 sources


A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
# print 'Shape of measurements: {}'.format(X.shape)
# Say we have 100 devices which can measure the combination of the
# 3 signals


nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
# print 'Recovered signal shape: {}'.format(S_.shape)
# prints (2000, 300)


pca = PCA(n_components=3)
H = pca.fit_transform(X)
models = [X, S, S_, H]
names = [
  'Observations (first 3 measurements',
  'True sources',
  'NMF recovered signals',
  'PCA recovered signals']
fig, axes = plt.subplots(
  4, figsize=(8, 4), gridspec_kw={'hspace': .5}, subplot_kw={'xticks': (), 'yticks': ()})
for model, name, ax in zip(models, names, axes):
  ax.set_title(name)
  ax.plot(model[:,:3], '-')
plt.show()
