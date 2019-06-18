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

from loader import load_data_svm

X_bow_train, X_bow_test, X_tfidf_train, X_tfidf_test, y_train, y_test = load_data_svm()

X_train = X_tfidf_train
X_test = X_tfidf_test

def objective_func(params):

    clf = SVC(**params)
    clf.fit(X_train,y_train)
    y_pred_train = clf.predict(X_train)

    loss = mean_squared_error(y_train,y_pred_train)

    score = cross_val_score(clf, X_train, y_train, cv=3).mean()
    print("Test Score:", clf.score(X_test,y_test))
    print("Train Score:", clf.score(X_train,y_train))
    print("CV Score:", score)
    print("=================\n")

    return loss

space = {'C': hp.choice('C', [0.001, 0.01, 0.1, 1, 10]),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'degree': hp.choice('degree',range(1,15)),
        'gamma':hp.choice('gamma',[0.001, 0.01, 0.1, 1])
        }

# Trials
trials = Trials()

# Set optimization algorithm
algo = partial(tpe.suggest, n_startup_jobs=10, gamma=0.25)

best = fmin(fn=objective_func,
        space=space,
        algo=algo,
        max_evals=100,
        trials=trials)

best_parameters = space_eval(space, best)
print("Best parameters:", str(best_parameters))

# Print time
tdiff = trials.trials[-1]['book_time'] - trials.trials[0]['book_time']
print("Time:" + str(tdiff.total_seconds() / 60))

# Set params
clf = SVC(C=best_parameters['C'], kernel=best_parameters['kernel'], gamma=best_parameters['gamma'], degree=best_parameters['degree'])

# Fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Predict
score = classification_report(y_pred, y_test)
print(score)
