import sys
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
ACH_feature =[]
Flex_feature = []
for seq in seqlist:
  ### ACH_feature
  temp_ACH = []
  ###Flex_feature
  temp_Flex = []
  for i in range(1,10,1):
    hydro_index_count = 0.0
    flex_index_count = 0.0
    for j in range(9-i,10+i,1):        
      hydro_index_count = hydro_index_count + hydro_index[acids.index(seq[j])]
      flex_index_count = flex_index_count + flex_index[acids.index(seq[j])]
    avg_hydro = hydro_index_count/(2*i+1)
    avg_flex = flex_index_count/(2*i+1)
    temp_ACH.append(round(avg_hydro,3))
    temp_Flex.append(round(avg_flex,3))
  ACH_feature.append(temp_ACH)
  Flex_feature.append(temp_Flex)
ACH = np.array(ACH_feature)
Flex = np.array(Flex_feature)
new_feature = np.hstack((ACH,Flex))
X = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
X = np.hstack((new_feature, X))
Y = np.load('/home/mlb2017/res/phosphosite/trainY.npy')
kf = KFold(len(X),n_folds=10,shuffle=True)
sum_train_score = 0
sum_test_score = 0
sum_auc1 = 0
f = open('Flex_window_result.txt','a')
f.write('python3 Flex_window.py'+ sys.argv[1] +' '+sys.argv[2])
f.write('\n')
for train_index,test_index in kf:
  train_X,test_X = X[train_index],X[test_index]
  train_Y,test_Y = Y[train_index],Y[test_index]
  clf = Pipeline([('scaler', StandardScaler()),
                  ('pca',PCA(n_components=100)),
                  ('clf', SVC(C=float(sys.argv[1]),gamma=float(sys.argv[2]),probability=True))])
  clf.fit(train_X, train_Y)
  sum_train_score = sum_train_score + clf.score(train_X,train_Y)
  sum_test_score = sum_test_score + clf.score(test_X,test_Y)
  print('Train Accuracy: {:.3f}'.format(clf.score(train_X,train_Y)))
  print('Test Accuracy: {:.3f}'.format(clf.score(test_X,test_Y)))
  Y_score = clf.predict_proba(test_X)[:, 1]  
  sum_auc1 = sum_auc1 + roc_auc_score(test_Y, Y_score)
  print('AUC: {:.3f}'.format(roc_auc_score(test_Y, Y_score)))
  f.write('Train Accuracy: {:.3f}'.format(clf.score(train_X,train_Y)))
  f.write('\n')
  f.write('Test Accuracy: {:.3f}'.format(clf.score(test_X,test_Y)))
  f.write('\n')
  f.write('AUC: {:.3f}'.format(roc_auc_score(test_Y, Y_score)))
  f.write('\n')
  f.write('---------------------------------------------------------------')
  f.write('\n')
  print('---------------------------------------------------------------')
print('Train Avg Accuracy: {:.3f}'.format(sum_train_score/10))
print('Test Avg Accuracy: {:.3f}'.format(sum_test_score/10))
print('AUC: {:.3f}'.format(sum_auc1/10))
f.write('Train Avg Accuracy: {:.3f}'.format(sum_train_score/10))
f.write('\n')
f.write('Test Avg Accuracy: {:.3f}'.format(sum_test_score/10))
f.write('\n')
f.write('AUC: {:.3f}'.format(sum_auc1/10))
f.write('\n')
