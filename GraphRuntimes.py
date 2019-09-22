import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 'ANN', 'Boosting', 'DT', 'KNN', 'SVM'
algs = ['DT', 'ANN', 'Boosting', 'SVM', 'KNN']
datasets = ['Credit Card Default', 'Wine']
pickles = []
train = []
test  = []
score = []

for alg in algs:
    for dataset in datasets:
        pickles.append('{}-{}.pickle'.format(alg, dataset))

for cucumber in pickles:
    pickle_in = open('Pickles/{}'.format(cucumber),"rb")
    cv_result_dict = pickle.load(pickle_in)
    print(cucumber)

    train = np.append(train, np.mean(cv_result_dict['mean_fit_time'])   )
    test  = np.append(test , np.mean(cv_result_dict['mean_score_time']) )
    score = np.append(score, np.max(cv_result_dict['mean_test_score'])  )

    print('train:  {}'.format(np.mean(cv_result_dict['mean_fit_time'])  ))
    print('test:   {}'.format(np.mean(cv_result_dict['mean_score_time'])))
    print('score:  {}'.format(np.max(cv_result_dict['mean_test_score']) ))


