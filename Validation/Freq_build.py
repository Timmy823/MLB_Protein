import sys
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
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
        acids_count[index] = acids_count[index] + 1
  for index in range(len(acids_count)):
    acids_count[index] = float(acids_count[index]/19)
  Freq_feature.append(acids_count)
  acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
ACH = np.array(ACH_feature)
Freq = np.array(Freq_feature)
new_feature =  np.hstack((ACH,Freq))
X = np.load('/home/mlb2017/res/phosphosite/trainX.npy')
train_X = np.hstack((new_feature, X))
train_Y = np.load('/home/mlb2017/res/phosphosite/trainY.npy')
clf = Pipeline([('scaler', StandardScaler()),
                ('pca',PCA(n_components=100)),
                ('clf', SVC(C=float(sys.argv[1]),gamma=float(sys.argv[2]),probability=True))])
clf.fit(train_X, train_Y)
print('Train Accuracy: {:.3f}'.format(clf.score(train_X,train_Y)))
f = open('./Freq_model.pkl', "wb")
pickle.dump(clf, f)

