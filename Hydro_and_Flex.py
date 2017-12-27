import sys
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

with open('/home/mlb2017/res/phosphosite/train_data') as f:
    data = f.readlines()
data = [x.strip() for x in data]

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
X = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
X = np.hstack((add_feature, X))
Y = np.load('/home/mlb2017/res/phosphosite/trainY.npy')
kf = KFold(len(X),n_folds=10,shuffle=True)
sum_train_score = 0
sum_test_score = 0
sum_auc1 = 0

for train_index,test_index in kf:
  train_X,test_X = X[train_index],X[test_index]
  train_Y,test_Y = Y[train_index],Y[test_index]
  clf = Pipeline([('scaler', StandardScaler()),
                ('clf', SVC(C=float(sys.argv[1]),gamma=float(sys.argv[2]),probability=True))])
  clf.fit(train_X, train_Y)
  sum_train_score = sum_train_score + clf.score(train_X,train_Y)
  sum_test_score = sum_test_score + clf.score(test_X,test_Y)
  print('Train Accuracy: {:.3f}'.format(clf.score(train_X,train_Y)))
  print('Test Accuracy: {:.3f}'.format(clf.score(test_X,test_Y)))
  Y_score = clf.predict_proba(test_X)[:, 1]  
  sum_auc1 = sum_auc1 + roc_auc_score(test_Y, Y_score)
  print('AUC: {:.3f}'.format(roc_auc_score(test_Y, Y_score)))
  print('---------------------------------------------------------------')
print('Train Avg Accuracy: {:.3f}'.format(sum_train_score/10))
print('Test Avg Accuracy: {:.3f}'.format(sum_test_score/10))
print('AUC: {:.3f}'.format(sum_auc1/10))


    

