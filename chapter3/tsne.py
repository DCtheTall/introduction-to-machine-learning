"""
Manifold Learning with t-SNE
----------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


digits = load_digits()
fig, axes = plt.subplots(
  2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits['images']):
  ax.imshow(img)
plt.show()
# Plot 10 handwritten digits


def plot_decomposed_data(decomposed_digits):
  """
  Plot decomposed digit data

  """
  colors = [
      '#476a2a', '#7851b8', '#bd3430', '#4a2d4e', '#875525',
      '#a83683', '#4e655e', '#853541', '#3a3120', '#535d8e',
  ]
  plt.figure(figsize=(10, 10))
  plt.xlim(decomposed_digits[:, 0].min(), decomposed_digits[:, 0].max())
  plt.ylim(decomposed_digits[:, 1].min(), decomposed_digits[:, 1].max())
  for i in range(len(digits['data'])):
    plt.text(
        decomposed_digits[i, 0],
        decomposed_digits[i, 1],
        str(digits['target'][i]),
        color=colors[digits['target'][i]],
        fontdict={'weight': 'bold', 'size': 9})
  plt.xlabel('First principal component')
  plt.ylabel('Second principal component')
  plt.show()


pca = PCA(n_components=2)
digits_pca = pca.fit_transform(digits['data'])
plot_decomposed_data(digits_pca)
# Applying PCA to classify the handwritten digits is not effective
# although it can distinguish between a few numbers, most overlap


tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits['data'])
plot_decomposed_data(digits_tsne)
# Manifold learning is much more effective and
# can classify the numbers unsupervised
