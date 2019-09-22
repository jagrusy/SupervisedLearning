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

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# print('reading data')
# df = pd.read_excel("./Data/ccdefault.xls", header=1)
# df = df.apply(pd.to_numeric, errors='coerce')
# print('printing data')
# print(df.head())

# # Include all columns but the index and the output
# X = df.iloc[1:,1:-2].values
# print(X[0:5])

# # Select last column
# y = df.iloc[1:,-1].values

# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=3)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

# clf = svm.SVC(kernel='rbf')
# clf.fit(X_train, y_train)
# yhat = clf.predict(X_test)
# yhat[0:5]

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
# np.set_printoptions(precision=2)

# print(classification_report(y_test, yhat))

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=['No Default(0)','Defaulted(1)'],normalize= False,  title='Confusion matrix')

# Compute Jaccard index and f1 score
# print('f1 score:')
# print(f1_score(y_test, yhat, average='weighted'))
# print('Jaccard Similarity Score:')
# print(jaccard_similarity_score(y_test, yhat))

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