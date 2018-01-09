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
    charge= [1,1,2,2,1,1,1,1,3,1,1,1,1,1,3,1,1,1,1,1]
    Hydro=[2,1,3,3,1,2,2,1,3,1,1,3,2,3,3,2,2,1,1,2]
    vanderWaals=[1,1,1,2,3,1,3,2,3,2,3,2,1,2,3,1,1,2,3,3]
    polarity=[2,1,3,3,1,2,3,1,3,1,1,3,2,3,3,2,2,1,1,1]
    polarizability=[1,2,1,2,3,1,3,2,3,2,3,2,2,2,3,1,1,2,3,3]
    secondstructure=[2,3,1,2,3,1,2,3,2,2,2,1,1,2,2,1,3,3,3,3]
    solventaccessibility=[1,1,3,3,1,1,2,1,3,1,2,3,2,3,3,2,2,1,1,2]
    encode=[]
    encode.append(charge)
    encode.append(Hydro)
    encode.append(vanderWaals)
    encode.append(polarity)
    encode.append(polarizability)
    encode.append(secondstructure)
    encode.append(solventaccessibility)
    
    featurelist = []
    acids_count = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    hydro_index = [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06, 0.64, -0.78,0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26]
    for seq in seqlist:
        temp=[]
        seqencode=[]
        for j in range(0,7,1):
            seqtemp=[]
            for item in seq:
                for index in range(len(acids)):
                    if item==acids[index]:
                        seqtemp.append(encode[j][index])
            seqencode.append(seqtemp)
            
        for k in range(0,7,1):
            #composition
            for i in range(1,4,1):
                count=seqencode[k].count(i) 
                count/=19
                temp.append(count)
            #transition
            n12=0    
            n21=0
            n23=0
            n32=0
            n13=0
            n31=0
            for m in range(len(seq)-1): #index:0~17
                if seqencode[k][m]==1 and seqencode[k][m+1]==2:
                    n12+=1
                elif seqencode[k][m]==2 and seqencode[k][m+1]==1:
                    n21+=1
                elif seqencode[k][m]==2 and seqencode[k][m+1]==3:
                    n23+=1
                elif seqencode[k][m]==3 and seqencode[k][m+1]==2:
                    n32+=1
                elif seqencode[k][m]==1 and seqencode[k][m+1]==3:
                    n13+=1
                elif seqencode[k][m]==3 and seqencode[k][m+1]==1:
                    n31+=1
            temp.append((n12+n21)/18)
            temp.append((n23+n32)/18)    
            temp.append((n13+n31)/18)
            #distribution
            for i in range(1,4,1):
                count=seqencode[k].count(i)
                if count==0:
                    for i in range(0,5,1):
                        temp.append(0)
                else:
                    onequarter=int(count/4) #25%位置
                    if onequarter==0:
                        onequarter=1
                    half=int(count/2)#50%位置
                    if half==0:
                        half=1
                    threequarter=int(count*3/4)#75%位置
                    if threequarter==0:
                        threequarter=1
                    locatecount=0
                    for m in range(len(seq)): 
                        if seqencode[k][m]==i: 
                            locatecount+=1
                            if locatecount==1: #first ditribution
                                temp.append((m+1)*100/19)
                            if locatecount==onequarter:#25% ditribution
                                temp.append((m+1)*100/19)
                            if locatecount==half:#50% ditribution
                                temp.append((m+1)*100/19)
                            if locatecount==threequarter:#75% ditribution
                                temp.append((m+1)*100/19)
                            if locatecount==count:#100% ditribution
                                temp.append((m+1)*100/19)                 
        for i in range(1,10,1):
            hydro_index_count = 0.0
            for j in range(9-i,10+i,1):        
                  hydro_index_count = hydro_index_count + hydro_index[acids.index(seq[j])]
            avg = hydro_index_count/(2*i+1)
            temp.append(round(avg,3))
       # print(len(temp))
        featurelist.append(temp)
    add_feature = np.array(featurelist)
    X = np.load('/home/mlb2017/res/phosphosite/'+data_string+'X.npy')
    X = np.hstack((add_feature, X))
    print(X.shape)
    Y = np.load('/home/mlb2017/res/phosphosite/'+data_string+'Y.npy')
    np.save( 'ctd'+data_string+'_X', X)
    np.save('ctd'+data_string+'_Y', Y)
    return X,Y
trainX,trainY=feature(data,'train')
validX,validY=feature(data2,'validation')

'''with open('Flex_result.txt', 'a') as f:          #write file
    f.write(time.strftime("%Y %b %d %H:%M:%S\nPCA=50\n"))
    #parameters = {'clf__C': [5],'clf__gamma': [0.01]}
    parameters={'clf__C': [0.5, 1, 2.5,5,10], 'clf__gamma': [0.025, 0.01, 0.005]}
    clf = Pipeline([('scaler', StandardScaler()),
                      ('pca',PCA(n_components=50)),
                      ('clf', SVC(probability=True))])
    scores = ['accuracy', 'roc_auc']
    for score in scores:
        clf = GridSearchCV(estimator=clf, 
                          param_grid=parameters, 
                          scoring='%s' % score, 
                          cv=10,
                          n_jobs=-1)

        clf.fit(trainX, trainY)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            f.write("%0.3f (+/-%0.03f) for %r\n"% (mean, std * 2, params))
        f.write('\n')
        f.write(json.dumps(clf.best_params_))
        f.write('\n')
        f.write('Train '+str(score)+' : {:.3f}'.format(clf.best_score_))
        f.write('\n')
        clf = clf.best_estimator_
        f.write('Test accuracy: {:.3f}'.format(clf.score(validX, validY)))
        f.write('\n')
        f.write('AUC: {:.3f}'.format(roc_auc_score(validY, clf.predict_proba(validX)[:, 1] )))
        f.write('\n\n====================================\n\n\n')'''