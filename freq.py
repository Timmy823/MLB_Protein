import sys
import time
import numpy as np
import sklearn
import json
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
trainX = np.load('freqpcatrainX.npy') 
trainY = np.load('freqpcatrainY.npy')
validX = np.load('freqpcavalidX.npy') 
validY = np.load('freqpcavalidY.npy')

with open('freqpca_result.txt', 'a') as f:          #write file
    f.write(time.strftime("%Y %b %d %H:%M:%S\nfreqpca_Feature,PCA=20+20\n"))
    #parameters = {'clf__C': [5],'clf__gamma': [0.01]}
    parameters={'clf__C': [0.5, 1, 2.5,5,10], 'clf__gamma': [0.025, 0.01, 0.005]}
    clf = Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(probability=True))])
    scores = ['roc_auc']
    for score in scores:
        clf = GridSearchCV(estimator=clf, 
                          param_grid=parameters, 
                          scoring='%s' % score, 
                          cv=10,
                          n_jobs=-1)

        clf.fit(trainX, trainY)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            f.write("%0.3f (+/-%0.03f) for %r\n"% (mean, std * 2, params))
        f.write('\n')
        f.write(json.dumps(clf.best_params_))
        f.write('\n')
        f.write('Train '+str(score)+' : {:.3f}'.format(clf.best_score_))
        f.write('\n')
        clf = clf.best_estimator_
        f.write('Test accuracy: {:.3f}'.format(clf.score(validX, validY)))
        f.write('\n')
        f.write('AUC: {:.3f}'.format(roc_auc_score(validY, clf.predict_proba(validX)[:, 1] )))
        f.write('\n\n====================================\n\n\n')