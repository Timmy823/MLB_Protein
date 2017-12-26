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
flex_index = [0.36,0.35,0.51,0.5,0.31,0.54,0.32,0.46,0.47,0.37,0.3,0.46,0.51,0.49,0.53,0.51,0.44,0.39,0.31,0.42]
feature_list = []
acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
flex_index_count = 0;
for seq in seqlist:
  temp = []
  for item in seq:
    for index in range(len(acids)):
      if item == acids[index] :
        acids_count[index] = acids_count[index] + 1
        flex_index_count = flex_index_count + flex_index[index] 
  for index in range(len(acids_count)):
    acids_count[index] = float(acids_count[index]/19)
  temp = acids_count
  temp.append(flex_index_count);
  feature_list.append(temp)
  acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
###print(freqlist)

train_X = np.array(feature_list)
train_Y = np.array(phoslist)
clf = Pipeline([('scaler', StandardScaler()),
                ('clf', SVC(C=float(sys.argv[1]),gamma=float(sys.argv[2])))])
clf.fit(train_X, train_Y)
scores = cross_val_score(clf,train_X,train_Y,cv=10,n_jobs=1)
print('Accuracy list: {}'.format(scores))
print('Test Accuracy Range: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))

        
    

