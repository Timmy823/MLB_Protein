import sys
import time
import numpy as np
import sklearn
import json
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

with open('/home/mlb2017/res/phosphosite/train_data') as f:
    data = f.readlines()
data = [x.strip() for x in data]
with open('/home/mlb2017/res/phosphosite/validation_data') as g:
    data2 = g.readlines()
data2 = [x.strip() for x in data2]

def feature(data,data_string):
        seqlist = []
        for site in data:
            site = site.split(' ')
            seqlist.append(site[2])
        ###print(seqlist)
        acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
        featurelist = []
        acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for seq in seqlist:
          temp = []
          for i in range(1,10,1):
            hydro_index_count = 0.0
            for j in range(9-i,10+i,1):        
              hydro_index_count = hydro_index_count + hydro_index[acids.index(seq[j])]
            avg = hydro_index_count/(2*i+1)
            temp.append(round(avg,3))

          featurelist.append(temp)
          acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        ###print(freqlist)
        add_feature = np.array(featurelist)
        X = np.load('/home/mlb2017/res/phosphosite/'+data_string+'X.npy')
        X = np.hstack((add_feature, X))
        Y = np.load('/home/mlb2017/res/phosphosite/'+data_string+'Y.npy')
        return X,Y
scaler = StandardScaler()

trainX,trainY=feature(data,'train')
validX,validY=feature(data2,'validation')
pca=PCA(n_components=20)
trainX=scaler.fit_transform(trainX)
validX=scaler.fit_transform(validX)
trainX=np.array(pca.fit_transform(trainX))
validX=np.array(pca.fit_transform(validX))
#with open('freqpca.txt', 'a') as f:
#     for pcaitem in pca.components_:
#        f.write("%s\n" % pcaitem)
def freqfeature(data,data_string):
        seqlist = []
        for site in data:
            site = site.split(' ')
            seqlist.append(site[2])
        ###print(seqlist)
        acids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
        featurelist = []
        acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for seq in seqlist:
            for item in seq:
                for index in range(len(acids)):
                    if item == acids[index] :
                        acids_count[index] = acids_count[index] + 1
            for index in range(len(acids_count)):
                acids_count[index] = float(acids_count[index]/19)
            featurelist.append(acids_count)
            acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        ###print(freqlist)
        add_feature = np.array(featurelist)
        return add_feature
train_freqX=freqfeature(data,'train')
valid_freqX=freqfeature(data2,'validation')
trainX = np.hstack((train_freqX,trainX)) 
validX = np.hstack((valid_freqX,validX))
print(trainX.shape)
print(validX.shape)
np.save("freqpcatrainX.npy",trainX)
np.save("freqpcatrainY.npy",trainY)
np.save("freqpcavalidX.npy",validX)
np.save("freqpcavalidY.npy",validY)