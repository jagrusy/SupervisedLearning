import pandas as pd
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_similarity_score
import itertools
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from util import getCreditCardData, getWineData, plot_learning_curve, save_cv


def SVM(X_train, X_test, y_train, y_test, data_name):
    # Train Model and Predict
    # param_grid = {"kernel" : ["sigmoid", "poly", "rbf"],
    #             "C" : [0.1, 0.5, 1.0, 1.5]
    #             }

    param_distributions = {"kernel" : ["sigmoid", "poly", "rbf"],
            "gamma" : np.linspace(0.001, 1.0, 1000)
            }
    
    # clf = svm.SVC(gamma='scale')
    clf = svm.SVC()

    # run grid search on dataset
    # grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search = RandomizedSearchCV(clf, param_distributions=param_distributions, cv=2, n_iter=20, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    best_params = grid_search.best_params_
    print(best_params)
    save_cv(grid_search.cv_results_, 'SVM', data_name)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    title = 'Learning Curves (SVM Classifier) - {}'.format(data_name)

    # estimator = svm.SVC(gamma='scale', **best_params)
    estimator = svm.SVC(**best_params)
    print('plotting learning curve for {}'.format(estimator))
    plot_learning_curve(estimator, title, X, y, ylim=(0.4, 1.01), cv=2, n_jobs=-1)
    plt.savefig('Figs/SVM-learningcurve-{}'.format(data_name))

if __name__ == "__main__":
    np.random.seed(0)
    test_size = 0.2

    X_train1, X_test1, y_train1, y_test1 = getCreditCardData(path='./Data/ccdefault.xls', test_size=0.1, train_size=0.4)
    X_train2, X_test2, y_train2, y_test2 = getWineData(path='./Data/winequality-white.csv', test_size=test_size)

    print(X_train1.shape)
    print(X_test1.shape)
    print(y_train1.shape)
    print(y_test1.shape)
    print(X_train2.shape)
    print(X_test2.shape)
    print(y_train2.shape)
    print(y_test2.shape)

    SVM(X_train1, X_test1, y_train1, y_test1, 'Credit Card Default')
    SVM(X_train2, X_test2, y_train2, y_test2, 'Wine')