
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from util import getCreditCardData, getWineData, plot_learning_curve, save_cv

def Boosting(X_train, X_test, y_train, y_test, data_name, lc_y_min=0.4, lc_y_max=1.01):
    # Train Model and Predict
    # dt1 = DecisionTreeClassifier(criterion="gini", max_depth=1)
    # dt2 = DecisionTreeClassifier(criterion="gini", max_depth=2)
    dt3 = DecisionTreeClassifier(criterion="gini", max_depth=3)
    # dt4 = DecisionTreeClassifier(criterion="gini", max_depth=4)
    # dt5 = DecisionTreeClassifier(criterion="gini", max_depth=5)
    param_grid = {"base_estimator": [dt3],
                "learning_rate" : np.linspace(0.5, 10.0, 20),
                "n_estimators": range(1, 200, 20)
                }

    clf = AdaBoostClassifier()

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    best_params = grid_search.best_params_
    print(best_params)
    save_cv(grid_search.cv_results_, 'Boosting', data_name)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    title = 'Learning Curves (Boosting Classifier) - {}'.format(data_name)

    estimator = AdaBoostClassifier( **best_params)
    print('plotting learning curve for {}'.format(estimator))
    plot_learning_curve(estimator, title, X, y, ylim=(lc_y_min, lc_y_max), cv=cv, n_jobs=4)
    plt.savefig('Figs/Boosting-learningcurve-{}'.format(data_name))

if __name__ == "__main__":
    np.random.seed(0)
    test_size = 0.2

    X_train1, X_test1, y_train1, y_test1 = getCreditCardData(path='./Data/ccdefault.xls', test_size=test_size)
    X_train2, X_test2, y_train2, y_test2 = getWineData(path='./Data/winequality-white.csv', test_size=test_size)

    Boosting(X_train1, X_test1, y_train1, y_test1, 'Credit Card Default', 0.8, 0.85)
    Boosting(X_train2, X_test2, y_train2, y_test2, 'Wine', 0.4, 1.01)
