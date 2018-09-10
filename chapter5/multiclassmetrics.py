"""
Metrics for Multiclass Classification
-------------------------------------

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  classification_report,
  f1_score,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


digits = load_digits()
X_train, X_test, y_train, y_test = \
  train_test_split(digits['data'], digits['target'], random_state=0)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
# print 'Accuracy: {:.3f}'.format(accuracy_score(y_test, pred))
# print 'Confusion matrix:\n{}'.format(confusion_matrix(y_test, pred))
# Accuracy and confusion matrix for using logistic regression
# to predict what number handwritten digits are


# scores_image = mglearn.tools.heatmap(
#   confusion_matrix(y_test, pred),
#   xlabel='Predicted label',
#   ylabel='True label',
#   xticklabels=digits['target_names'],
#   yticklabels=digits['target_names'],
#   cmap=plt.cm.gray_r,
#   fmt='%d',
# )
# plt.title('Confusion matrix')
# plt.gca().invert_yaxis()
# plt.show()
# Plots heatmap of the confusion matrix for this classification model


# print classification_report(y_test, pred)
# Prints the classification report for the model


print 'Micro average f1 score: {:.3f}'.format(
  f1_score(y_test, pred, average='micro'))
print 'Macro average f1 score: {:.3f}'.format(
  f1_score(y_test, pred, average='macro'))
# Prints the micro and macro average f1 score
# The micro score computes the aggregated precision and recall across all classes
# The macro score computes the average of the unweighted per-class f1 score
