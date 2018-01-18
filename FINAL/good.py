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
acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

code = []
acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
Hydro = [2,1,3,3,1,2,2,1,3,1,1,3,2,3,3,2,2,1,1,2]
polarity = [2,1,3,3,1,2,3,1,3,1,1,3,2,3,3,2,2,1,1,1]
flex = [1,1,3,3,1,3,1,2,2,1,1,2,3,2,3,3,2,1,1,2]
code.append(Hydro)
code.append(polarity)
code.append(flex)
ACH_feature =[]
Freq_feature =[]
new_feature =[]
for seq in seqlist:
  ### ACH_feature
  register = []
  for i in range(1,10,1):
    hydro_index_count = 0.0
    for j in range(9-i,10+i,1):
      hydro_index_count = hydro_index_count + hydro_index[acids.index(seq[j])]
    avg = hydro_index_count/(2*i+1)
    register.append(round(avg,3))
  ACH_feature.append(register)
  #Frequency
  for item in seq:
    for index in range(len(acids)):
      if item == acids[index] :
        acids_count[index] = acids_count[index] + 1
  for index in range(len(acids_count)):
    acids_count[index] = float(acids_count[index]/19)
  Freq_feature.append(acids_count)
  acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
  #CTD
  temp = []
  new_temp = []
  encode = []
  temp_code = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #len = 19 
  count = [0,0,0]
  #encoding
  for i in range(3):
    for item in range(len(seq)):
      for index in range(len(acids)):
        if seq[item] == acids[index] :
          temp_code[item] = code[i][index]
    new_code = temp_code[:]
    encode.append(new_code)
  for i in range(3):  
    #composition [percentage*3 dimen]
    composition = [0.0,0.0,0.0]
    for order in range(len(encode[i])):
      count[encode[i][order]-1] += 1
    for i in range(3):
      composition[i] = round(count[i]/19,3)
    #transition
    N_count = [0,0,0,0,0,0] #[n12,n21,n23,n32,n13,n31]
    transition= [0.0,0.0,0.0]
    for order in range(len(encode[i])-1):
      if encode[i][order] == 1 and encode[i][order+1] == 2 :
        N_count[0] += 1
      elif encode[i][order] == 2 and encode[i][order+1] == 1 :
        N_count[1] += 1
      elif encode[i][order] == 2 and encode[i][order+1] == 3 :
        N_count[2] += 1
      elif encode[i][order] == 3 and encode[i][order+1] == 2 :
        N_count[3] += 1
      elif encode[i][order] == 1 and encode[i][order+1] == 3 :
        N_count[4] += 1
      elif encode[i][order] == 3 and encode[i][order+1] == 1 :
        N_count[5] += 1
    for i in range(3):  
      transition[i] = round((N_count[2*i]+N_count[2*i+1])/18,3)
    #distibution
    distribution = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    target = [0,0,0,0,0,0,0,0,0]
    point = [0,0,0]
    for i in range(3):
      target[3*i] = round(count[i]/4,0)
      target[3*i+1] = round(count[i]/2,0)
      target[3*i+2] = round(count[i]*3/4,0)
    for order in range(len(encode[i])):
      if encode[i][order] == 1:
        point[0] += 1
        if point[0] == 1:
          distribution[0] = round((order+1)/19,3)
        if point[0] == count[0]:
          distribution[4] = round((order+1)/19,3)
        for m in range(3):
          if point[0] == target[m]:
            distribution[m+1] = round((order+1)/19,3)
      elif encode[i][order] == 2:
        point[1] += 1
        if point[1] == 1:
          distribution[5] = round((order+1)/19,3)
        if point[1] == count[1]:
          distribution[9] = round((order+1)/19,3)
        for m in range(3):
          if point[1] == target[3+m]:
            distribution[m+6] = round((order+1)/19,3)
      elif encode[i][order] == 3:
        point[2] += 1
        if point[2] == 1:
          distribution[10] = round((order+1)/19,3)
        if point[2] == count[2]:
          distribution[14] = round((order+1)/19,3)
        for m in range(3):
          if point[2] == target[6+m]:
            distribution[m+11] = round((order+1)/19,3)
    temp = composition + transition + distribution
    new_temp = new_temp + temp
  new_feature.append(new_temp)
ACH = np.array(ACH_feature)
Freq = np.array(Freq_feature)
new_feature = np.array(new_feature)
X = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
X = np.hstack((ACH, X))
new_feature = np.hstack((Freq, new_feature))
#feature selection
pca = PCA(n_components=100)
X = pca.fit_transform(X)
X = np.hstack((new_feature, X))
Y = np.load('/home/mlb2017/res/phosphosite/trainY.npy')
kf = KFold(len(X),n_folds=10,shuffle=True)
sum_train_score = 0
sum_test_score = 0
sum_auc1 = 0
f = open('result.txt','a')
f.write('python3 last.py '+ sys.argv[1] +' '+sys.argv[2])
f.write('\n')
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

