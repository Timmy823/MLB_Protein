import sys
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

with open('/home/mlb2017/res/phosphosite/train_data') as f:
    data = f.readlines()
data = [x.strip() for x in data]

seqlist = []
phoslist = []
for site in data:
    site = site.split(' ')
    seqlist.append(site[2])
    phoslist.append(site[0])
###print(seqlist)
acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
freqlist = [];
acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
for seq in seqlist:
  temp = []
  for item in seq:
    for index in range(len(acids)):
      if item == acids[index] :
        acids_count[index] = acids_count[index] + 1
  for index in range(len(acids_count)):
    acids_count[index] = float(acids_count[index]/19)
  freqlist.append(acids_count);
  acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
###print(freqlist)
train_X = np.array(freqlist)
train_Y = np.array(phoslist)
clf = Pipeline([('scaler', StandardScaler()),
                ('clf', SVC(C=float(sys.argv[1]),gamma=float(sys.argv[2])))])
clf.fit(train_X, train_Y)
scores = cross_val_score(clf,train_X,train_Y,cv=10,n_jobs=1)
print('Test Accuracy: {}'.format(scores))
print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))

        
    

