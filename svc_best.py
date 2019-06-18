import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, train_test_split)
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, hp, fmin
from sklearn.metrics import mean_squared_error
import torch

from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval

from data_loader import load_data_svm

X_bow_train, X_bow_test, X_tfidf_train, X_tfidf_test, y_train, y_test = load_data_svm()

X_train = X_tfidf_train
X_test = X_tfidf_test

# Set params
clf = SVC(C=10, kernel='linear', gamma=0.1, degree=1)
clf = SVC()

# Fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Predict
score = classification_report(y_pred, y_test)
print(score)
