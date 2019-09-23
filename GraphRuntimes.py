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

    if 'SVM' in cucumber:
        rbf_score = []
        poly_score = []
        sig_score = []
        rbf_time = []
        poly_time = []
        sig_time = []
        for i in range(len(cv_result_dict['mean_test_score'])):
            if cv_result_dict['param_kernel'][i] == 'rbf':
                rbf_score = np.append(rbf_score, cv_result_dict['mean_test_score'][i])
                rbf_time = np.append(rbf_time, cv_result_dict['mean_fit_time'][i])
            elif cv_result_dict['param_kernel'][i] == 'poly':
                poly_score = np.append(poly_score, cv_result_dict['mean_test_score'][i])
                poly_time = np.append(poly_time, cv_result_dict['mean_fit_time'][i])
            else:
                sig_score = np.append(sig_score, cv_result_dict['mean_test_score'][i])
                sig_time = np.append(sig_time, cv_result_dict['mean_fit_time'][i])

        print('rbf: score:{}->train time {}s'.format(np.max(rbf_score), np.mean(rbf_time )))
        print('poly: score:{}->train time {}s'.format(np.max(poly_score), np.mean(poly_time)))
        print('sig:score:{}->train time {}s'.format(np.max(sig_score), np.mean(sig_time )))


