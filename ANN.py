from sklearn.neural_network import MLPClassifier
import pylab as pl
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_similarity_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from util import getCreditCardData, getWineData, plot_learning_curve, save_cv

def ANN(X_train, X_test, y_train, y_test, data_name, lc_y_min=0.4, lc_y_max=1.01):
    # Train Model and Predict
    unique_vals = len(np.unique(y_test))

    clf = MLPClassifier(solver='sgd')


    param_grid = {
        "hidden_layer_sizes" : [(10,)],
        "alpha" : np.linspace(0.0001, 0.5, 50),
        "momentum" : np.linspace(0.1, 1.0, 10)
    }
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    best_params = grid_search.best_params_
    best_params = { **best_params }
    save_cv(grid_search.cv_results_, 'ANN', data_name)

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    title = 'Learning Curves (ANN Classifier) - {}'.format(data_name)

    estimator = MLPClassifier(solver='sgd',**best_params)
    print('plotting learning curve for {}'.format(estimator))
    plot_learning_curve(estimator, title, X, y, ylim=(lc_y_min, lc_y_max), cv=cv, n_jobs=4)
    plt.savefig('Figs/ANN-learningcurve-{}'.format(data_name))

if __name__ == "__main__":
    np.random.seed(0)
    test_size = 0.2

    X_train1, X_test1, y_train1, y_test1 = getCreditCardData(path='./Data/ccdefault.xls', test_size=test_size)
    X_train2, X_test2, y_train2, y_test2 = getWineData(path='./Data/winequality-white.csv', test_size=test_size)

    ANN(X_train1, X_test1, y_train1, y_test1, 'Credit Card Default', 0.75, 0.95)
    ANN(X_train2, X_test2, y_train2, y_test2, 'Wine', 0.4, 1.01)