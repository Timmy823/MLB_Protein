import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
# load model
f = open('./Flex_window_model.pkl', 'rb')
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
X = np.load('/home/mlb2017/res/phosphosite/validationX.npy')
test_X = np.hstack((new_feature, X))
test_Y = np.load('/home/mlb2017/res/phosphosite/validationY.npy')
print('Test Accuracy: {:.3f}'.format(clf.score(test_X,test_Y)))
Y_score = clf.predict_proba(test_X)[:, 1]
print('AUC: {:.3f}'.format(roc_auc_score(test_Y, Y_score)))
