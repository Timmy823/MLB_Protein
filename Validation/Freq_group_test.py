import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
# load model
f = open('./Freq_group_model.pkl', 'rb')
clf = pickle.load(f)
with open('/home/mlb2017/res/phosphosite/validation_data') as f:
    data = f.readlines()
data = [x.strip() for x in data]

seqlist = []
for site in data:
    site = site.split(' ')
    seqlist.append(site[2])
###print(seqlist)
acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
encodes = ["a","b","c","c","a","a","d","a","d","a","a","b","a","b","d","b","b","a","a","b"]
hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
acids_count = [0.0,0.0,0.0,0.0]
ACH_feature =[]
Freq_feature =[]
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
  for item in seq:
    for index in range(len(acids)):
      if item == acids[index] :
        if encodes[index] == 'a':
          acids_count[0] = acids_count[0] + 1
        if encodes[index] == 'b':
          acids_count[1] = acids_count[1] + 1
        if encodes[index] == 'c':
          acids_count[2] = acids_count[2] + 1
        if encodes[index] == 'd':
          acids_count[3] = acids_count[3] + 1
  for index in range(len(acids_count)):
    acids_count[index] = float(acids_count[index]/19)
  Freq_feature.append(acids_count)
  acids_count = [0.0,0.0,0.0,0.0]
ACH = np.array(ACH_feature)
Freq = np.array(Freq_feature)
new_feature =  np.hstack((ACH,Freq))
X = np.load('/home/mlb2017/res/phosphosite/validationX.npy')
test_X = np.hstack((new_feature, X))
test_Y = np.load('/home/mlb2017/res/phosphosite/validationY.npy')
print('Test Accuracy: {:.3f}'.format(clf.score(test_X,test_Y)))
Y_score = clf.predict_proba(test_X)[:, 1]
print('AUC: {:.3f}'.format(roc_auc_score(test_Y, Y_score)))

