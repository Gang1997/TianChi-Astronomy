# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:43:12 2018

@author: Administrator
"""
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
class Ensemble:
    def __init__(self,baseNum=10):
        self.baseNum=baseNum
    def fit(self,train):
        self.bases=[]
        for i in range(self.baseNum):            
#            print("base"+str(i))
            rus=RandomUnderSampler(ratio={"star":60000,"unknown":10000})
            x,y=rus.fit_sample(train.iloc[:,0:-1],train.iloc[:,-1])
            xg=XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100, silent=True, 
                 objective="multi:softprob", nthread=8, gamma=0.1, min_child_weight=1, 
                 max_delta_step=0, subsample=0.7, colsample_bytree=0.8, colsample_bylevel=1, 
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=2.5, base_score=0.5, seed=0, 
                 missing=None)
            xg.fit(X=x, y=y, sample_weight=None, eval_set=None, eval_metric="auc",
                   early_stopping_rounds=None, verbose=True)
            
            self.bases.append(xg)
    def predict(self,X):
        resProb=None
        resFinal=[]#最终结果
        labels=["galaxy","qso","star","unknown"]
        for m in self.bases:
            if(resProb is None):
                resProb=m.predict_proba(np.array(X))
            else:
                resProb+=m.predict_proba(np.array(X))
        #获取最终结果
        resIns=list(np.argmax(resProb,axis=1))
        for i in resIns:
            resFinal.append(labels[i])
        return resFinal
    
    
    
    
    
            