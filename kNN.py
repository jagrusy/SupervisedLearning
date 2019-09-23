import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
import time
import pickle

from util import getCreditCardData, getWineData, plot_learning_curve, save_cv

'''
Data Standardization give data zero mean and unit variance, it is good practice, 
especially for algorithms such as KNN which is based on distance of cases:
'''
def kNN(X_train, X_test, y_train, y_test, data_name):
    # Train Model and Predict
    Ks = 25
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))
    performance = {}
    performance['mean_fit_time'] = np.zeros((Ks-1))
    performance['mean_score_time'] = np.zeros((Ks-1))
    performance['mean_test_score'] = np.zeros((Ks-1))
    for n in range(1, Ks):
        # Train Model and Predict 
        train_start = time.time() 
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
        train_end = time.time()
        yhat = neigh.predict(X_test)
        test_end = time.time()
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
        std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])
        performance['mean_fit_time'][n-1] = train_end - train_start
        performance['mean_score_time'][n-1] = test_end - train_end
        performance['mean_test_score'] = metrics.accuracy_score(y_test, yhat)

    plt.title('Parameter Plot - Values for K - {}'.format(data_name))
    plt.plot(range(1,Ks),mean_acc,'g')
    plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.savefig('Figs/KNN-param-plot-{}'.format(data_name))
    plt.clf()
    save_cv(performance, 'KNN', data_name)


    print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
    print( "The best with K<10 was", mean_acc[0:9].max(), "with k=", mean_acc[0:9].argmax()+1)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    title = 'Learning Curves (kNN Classifier) - {}'.format(data_name)

    estimator = KNeighborsClassifier(n_neighbors=mean_acc.argmax()+1)
    print('plotting learning curve for {}'.format(estimator))
    plot_learning_curve(estimator, title, X, y, ylim=(0.4, 1.01), cv=cv, n_jobs=4)
    plt.savefig('Figs/KNN-learningcurve-{}'.format(data_name))
    plt.clf()

if __name__ == "__main__":
    np.random.seed(0)
    test_size = 0.2

    X_train1, X_test1, y_train1, y_test1 = getCreditCardData(path='./Data/ccdefault.xls', test_size=test_size)
    X_train2, X_test2, y_train2, y_test2 = getWineData(path='./Data/winequality-white.csv', test_size=test_size)

    kNN(X_train1, X_test1, y_train1, y_test1, 'Credit Card Default')
    kNN(X_train2, X_test2, y_train2, y_test2, 'Wine')