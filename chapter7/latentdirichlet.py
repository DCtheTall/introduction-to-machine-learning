"""
Latent Dirichlect Allocation
----------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


reviews_train = load_files('./aclImdb/train/')
text_train, y_train = reviews_train.data, reviews_train.target
text_train, y_train = \
    zip(*filter(lambda (X, y): y < 2, zip(text_train, y_train)))
repl_br_tags = lambda text: [doc.replace(b'<br />', b' ') for doc in text]
text_train = repl_br_tags(text_train)
reviews_test = load_files('./aclImdb/test/')
text_test, y_test = reviews_test.data, reviews_test.target
text_test = repl_br_tags(text_test)
# Code from previous section


vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)
lda = LatentDirichletAllocation(
  n_topics=10, learning_method='batch', max_iter=25, random_state=0)
document_topics = lda.fit_transform(X)
print 'lda.components_.shape: {}'.format(lda.components_.shape)
# Prints the shape of the components of the LDA, there are
# (num topics * num words) components, each one is a vector
# representation of each topic where the components are
# the frequency of each word in each topic


sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(
  topics=range(10),
  feature_names=feature_names,
  sorting=sorting,
  topics_per_chunk=5,
  n_words=10,
)
# Prints the 10 topics it divided the reviews into
# and the 10 most frequent words in each topic
# Some topics do seem to be genre specific though
# some also seem to be split based on the sentiment
# of the review


lda100 = LatentDirichletAllocation(
  n_topics=100,
  learning_method='batch',
  max_iter=25,
  random_state=0,
)
document_topics100 = lda100.fit_transform(X)
topics = np.array(
  [7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(
  topics=topics,
  feature_names=feature_names,
  sorting=sorting,
  topics_per_chunk=5,
  n_words=20,
)


music = np.argsort(document_topics100[:, 45])[::-1]
for i in music[:10]:
  print b''.join(text_train[i].split(b'.')[:2]) + b'\n'
# Prints positive movies about music, showing how


fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = [
  '{:>2}'.format(i) + ' '.join(words)
  for i, words in enumerate(feature_names[sorting[:, :2]])
]
for col in [0, 1]:
  start = col * 50
  end = (col + 1) * 50
  ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
  ax[col].set_yticks(np.arange(50))
  ax[col].set_yticklabels(topic_names[start:end], ha='left', va='top')
  ax[col].invert_yaxis()
  ax[col].set_xlim(0, 2000)
  yax = ax[col].get_yaxis()
  yax.set_tick_params(pad=130)
plt.tight_layout()
plt.show()
# Plot how important each topic was to the classifier
