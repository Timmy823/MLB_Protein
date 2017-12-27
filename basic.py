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
freqlist = []
acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for seq in seqlist:
  temp = []
  for item in seq:
    for index in range(len(acids)):
      if item == acids[index] :
        acids_count[index] = acids_count[index] + 1
  for index in range(len(acids_count)):
    acids_count[index] = float(acids_count[index]/19)
  freqlist.append(acids_count)
  acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
###print(freqlist)
freq = np.array(freqlist)
X = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
X = np.hstack((freq, X))
Y = np.load('/home/mlb2017/res/phosphosite/trainY.npy')
kf = KFold(len(X),n_folds=10,shuffle=True)
sum_train_score = 0
sum_test_score = 0
sum_auc1 = 0
###sum_auc2 = 0

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
  ###Y_decision = clf.decision_function(test_X)
  sum_auc1 = sum_auc1 + roc_auc_score(test_Y, Y_score)
  ###sum_auc2 = sum_auc2 + roc_auc_score(test_Y, Y_decision)
  print('AUC: {:.3f}'.format(roc_auc_score(test_Y, Y_score)))
  ###print('AUC(2): {:.3f}'.format(roc_auc_score(test_Y, Y_decision)))
  print('---------------------------------------------------------------')
print('Train Avg Accuracy: {:.3f}'.format(sum_train_score/10))
print('Test Avg Accuracy: {:.3f}'.format(sum_test_score/10))
print('AUC: {:.3f}'.format(sum_auc1/10))
###print('AUC(2): {:.3f}'.format(sum_auc2/10))


    

