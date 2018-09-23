"""
Advanced Tokenization, Stemming,
and Lemmatization
-----------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_files
import spacy
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression


POWERS_OF_TEN = [10 ** i for i in range(-3, 3)]


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


en_nlp = spacy.load('en')
stemmer = nltk.stem.PorterStemmer()

def compare_normalizations(doc):
  """
  Compare normalization from lemmatization (en_nlp)
  versus stemming (stemmer.stem)

  """
  doc_spacy = en_nlp(doc)
  print 'Lemmatization:\n{}'.format(
    [token.lemma_ for token in doc_spacy])
  print 'Stemming:\n{}'.format(
    [stemmer.stem(token.norm_.lower()) for token in doc_spacy])

# compare_normalizations(
#   u'Our meeting today was worse than yesterday, '
#   'I\'m scared of meeting the clients tomorrow.')
# Comparing the normalization process done by lemmatization,
# i.e. reducing it to tokens based on their context in the sentence,
# versus stemming, which is tokenizing by the root of the word


regexp = re.compile('(?u)\\b\\w\\w+\\b') # reg exp used by CounterVectorizer
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: \
  old_tokenizer.tokens_from_list(regexp.findall(string))

def custom_tokenizer(document):
  """
  Custom tokenizer for document processing pipeline

  """
  doc_spacy = en_nlp(document)
  return [token.lemma_ for token in doc_spacy]

lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
X_train_lemma = lemma_vect.fit_transform(text_train)
# print 'X_train_lemma.shape: {}'.format(X_train_lemma.shape)
X_train = CountVectorizer(min_df=5).fit_transform(text_train)
# print 'X_train.shape: {}'.format(X_train.shape)
# Lemmatization reduces the number of features taken into consideration
# X_train_lemma.shape: (25000, 21558)
# X_train.shape: (25000, 27271)


param_grid = {'C': POWERS_OF_TEN}
cv = StratifiedShuffleSplit(
  n_splits=5, test_size=0.99, train_size=0.01, random_state=0)
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print 'Best cross-validation score (standard CountVectorizer): {:.3f}'.format(grid.best_score_)
grid.fit(X_train_lemma, y_train)
print 'Best cross-validation score (lemmatization): {:.3f}'.format(grid.best_score_)
# Here we observe a modest boost in performance even when training with a small dataset
# when we use lemmatization for tokenization of words instead of using the literal string
# value
# We see it adds a modest boost to performance
