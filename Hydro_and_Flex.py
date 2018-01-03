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

with open('/home/mlb2017/res/phosphosite/train_data') as f:
    data = f.readlines()
data = [x.strip() for x in data]
with open('/home/mlb2017/res/phosphosite/validation_data') as g:
    data2 = g.readlines()
data2 = [x.strip() for x in data2]
def feature(data,data_string):
    seqlist = []
    for site in data:
        site = site.split(' ')
        seqlist.append(site[2])
    ###print(seqlist)
    acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
    flex_index = [0.36,0.35,0.51,0.5,0.31,0.54,0.32,0.46,0.47,0.37,0.3,0.46,0.51,0.49,0.53,0.51,0.44,0.39,0.31,0.42]
    featurelist = []
    acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    hydro_index_count = 0
    flex_index_count = 0
    for seq in seqlist:
      temp = []
      for item in seq:
        for index in range(len(acids)):
          if item == acids[index] :
            acids_count[index] = acids_count[index] + 1
            hydro_index_count = hydro_index_count + hydro_index[index] 
            flex_index_count = flex_index_count + flex_index[index]
      for index in range(len(acids_count)):
        acids_count[index] = float(acids_count[index]/19)
      temp = acids_count
      temp.append(hydro_index_count)
      temp.append(flex_index_count)
      featurelist.append(temp)
      acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    ###print(freqlist)
    add_feature = np.array(featurelist)
    X = np.load('/home/mlb2017/res/phosphosite/'+data_string+'X.npy')
    X = np.hstack((add_feature, X))
    Y = np.load('/home/mlb2017/res/phosphosite/'+data_string+'Y.npy')
    return X,Y
trainX,trainY=feature(data,'train')
validX,validY=feature(data2,'validation')

with open('Hydro&Flex_result.txt', 'a') as f:          #write file
    f.write(time.strftime("%Y %b %d %H:%M:%S\nPCA=50\n"))
    #parameters = {'clf__C': [5],'clf__gamma': [0.01]}
    parameters={'clf__C': [0.5, 1, 2.5,5,10], 'clf__gamma': [0.025, 0.01, 0.005]}
    clf = Pipeline([('scaler', StandardScaler()),
                      ('pca',PCA(n_components=50)),
                      ('clf', SVC(probability=True))])
    scores = ['accuracy', 'roc_auc']
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




    

