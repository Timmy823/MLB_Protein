import sys
import numpy as np
import sklearn
import json
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

with open('/home/mlb2017/res/phosphosite/train_data') as f: #train_data
    data = f.readlines()
data = [x.strip() for x in data]
seqlist = []
for site in data:
    site = site.split(' ')
    seqlist.append(site[2])

acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
ACH_feature =[]
for seq in seqlist:
  ### ACH_feature
  temp = []
  for i in range(1,10,1):
    hydro_index_count = 0.0
    for j in range(9-i,10+i,1):        
      hydro_index_count = hydro_index_count + hydro_index[acids.index(seq[j])]
    avg = hydro_index_count/(2*i+1)
    temp.append(round(avg,3))
  ACH_feature.append(temp)
ACH = np.array(ACH_feature)
trainX = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
trainX = np.hstack((ACH, trainX))
trainY = np.load('/home/mlb2017/res/phosphosite/trainY.npy')

with open('/home/mlb2017/res/phosphosite/validation_data') as f:#valid_data
    data = f.readlines()
data = [x.strip() for x in data]
seqlist = []
for site in data:
    site = site.split(' ')
    seqlist.append(site[2])

acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
ACH_feature =[]
for seq in seqlist:
  ### ACH_feature
  temp = []
  for i in range(1,10,1):
    hydro_index_count = 0.0
    for j in range(9-i,10+i,1):        
      hydro_index_count = hydro_index_count + hydro_index[acids.index(seq[j])]
    avg = hydro_index_count/(2*i+1)
    temp.append(round(avg,3))
  ACH_feature.append(temp)
ACH = np.array(ACH_feature)
validX = np.load('/home/mlb2017/res/phosphosite/validationX.npy')
validX = np.hstack((ACH, validX))
validY = np.load('/home/mlb2017/res/phosphosite/validationY.npy')

with open('result.txt', 'a') as f:          #write file
    f.write('PCA=50\n')
    #parameters = {'clf__C': [5],'clf__gamma': [0.01]}
    parameters = {'clf__C': [0.5, 1, 2.5,5,10], 'clf__gamma': [0.025, 0.01, 0.005]}
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
        f.write('\n\n====================================\n')
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
