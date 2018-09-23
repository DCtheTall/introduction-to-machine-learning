"""
Examining the Bag of Words Representation
on a Toy Dataset
----------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer


bards_words = [
    'The fool doth think he is wise,',
    'but the wise man knows himself to be a fool',
]
vect = CountVectorizer().fit(bards_words)
# print 'Vocabulary size: {}'.format(len(vect.vocabulary_))
# print 'Vocabulary content: {}'.format(vect.vocabulary_)
# Splits the text list into a vocabulary of 13 words


bag_of_words = vect.transform(bards_words)
# print 'bag_of_words: {}'.format(repr(bag_of_words))
# Transforms the text list into a SciPy sparse an
# m * n matrix where
#  m = number of entries in text list
#  n = number of words in the vocabulary


# print 'Dense representation of bag_of_words:\n{}'.format(
#     bag_of_words.toarray())
# Prints vector representation of each of the elements
# of the text list where the basis for the vector space
# is the vocabulary in bag of words


cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
# print 'Vocabulary size: {}'.format(len(cv.vocabulary_))
# print 'Vocabulary:\n{}'.format(cv.get_feature_names())
# The default behavior is to create one feature per
# term that is one token long, or one per unigram


cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
# print 'Vocabulary size: {}'.format(len(cv.vocabulary_))
# print 'Transformed data (dense):\n{}'.format(
#     cv.transform(bards_words).toarray())
# print 'Vocabulary:\n{}'.format(cv.get_feature_names())
# Here we count bigrams as one feature and discard unigrams
# We also see no bigram occurs more than once


cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
# print 'Vocabulary size: {}'.format(len(cv.vocabulary_))
# print 'Vocabulary:\n{}'.format(cv.get_feature_names())
# In practice you want to use a range from 1 up to 5
