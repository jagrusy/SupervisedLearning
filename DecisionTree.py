import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from util import getCreditCardData, getWineData, plot_learning_curve, save_cv

def DT(X_train, X_test, y_train, y_test, data_name, lc_y_min=0.4, lc_y_max=1.01):
    # Train Model and Predict
    param_grid = {"criterion" : ["gini", "entropy"],
                "max_depth" : range(1,50),
                }

    clf = DecisionTreeClassifier()

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    best_params = grid_search.best_params_
    print(best_params)
    save_cv(grid_search.cv_results_, 'DT', data_name)

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    title = 'Learning Curves (DT Classifier) - {}'.format(data_name)

    estimator = DecisionTreeClassifier(**best_params)
    print('plotting learning curve for {}'.format(estimator))
    plot_learning_curve(estimator, title, X, y, ylim=(lc_y_min, lc_y_max), cv=cv, n_jobs=4)
    plt.savefig('Figs/DT-learningcurve-{}'.format(data_name))
    plt.clf()

    # Plot param tuning
    n = 26
    test_mean_acc1 = np.zeros((n-1))
    test_std_acc1 = np.zeros((n-1))
    train_mean_acc1 = np.zeros((n-1))
    train_std_acc1 = np.zeros((n-1))
    for n in range(1, n):
        # Train Model and Predict
        print('Max depth: ', n) 
        tree = DecisionTreeClassifier(criterion="gini", max_depth = n)
        tree.fit(X_train, y_train)
        y_hat = tree.predict(X_test)
        y_hat_train = tree.predict(X_train)
        test_mean_acc1[n-1] = metrics.accuracy_score(y_test, y_hat)
        test_std_acc1[n-1] = np.std(y_hat == y_test)/np.sqrt(y_hat.shape[0])
        train_mean_acc1[n-1] = metrics.accuracy_score(y_train, y_hat_train)
        train_std_acc1[n-1] = np.std(y_hat_train == y_train)/np.sqrt(y_hat_train.shape[0])

    plt.plot(range(1, n+1), test_mean_acc1, 'r')
    plt.fill_between(range(1, n+1),test_mean_acc1 - 1 * test_std_acc1, test_mean_acc1 + 1 * test_std_acc1, alpha=0.10)
    plt.plot(range(1, n+1), train_mean_acc1, 'm')
    plt.fill_between(range(1, n+1),train_mean_acc1 - 1 * train_std_acc1, train_mean_acc1 + 1 * train_std_acc1, alpha=0.10)
    plt.legend(('Test Accuracy - {}'.format(data_name), 'Training Accuracy - {}'.format(data_name)))
    plt.ylabel('Accuracy')
    plt.xlabel('Decision Tree Depth')
    plt.tight_layout()
    plt.savefig('Figs/DT-depth-{}'.format(data_name))
    plt.clf()
if __name__ == "__main__":
    np.random.seed(0)
    test_size = 0.2

    X_train1, X_test1, y_train1, y_test1 = getCreditCardData(path='./Data/ccdefault.xls', test_size=test_size)
    X_train2, X_test2, y_train2, y_test2 = getWineData(path='./Data/winequality-white.csv', test_size=test_size)

    DT(X_train1, X_test1, y_train1, y_test1, 'Credit Card Default', 0.8, 0.9)
    DT(X_train2, X_test2, y_train2, y_test2, 'Wine', 0.4, 1.01)
