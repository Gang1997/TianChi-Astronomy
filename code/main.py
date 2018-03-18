# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:50:38 2018

@author: Administrator
"""
import numpy as np
import datetime
import pandas as pd
import dataPreProcessing as dpp
from sklearn.externals import joblib
import Ensemble
# In[] 合并数据集
#合并训练集
sourceFilePaths=[]
fileNames=pd.read_csv("../data/first_train_index_20180131.csv").sample(frac=1).reset_index(drop=True)
for i in list(fileNames.iloc[:,0]):
    sourceFilePaths.append("../data/first_train_data_20180131/"+str(i)+".txt")
dpp.mergeData(sourceFilePaths=sourceFilePaths,saveDir="../data",batchSize=4839,saveFileBaseName="train_",labels=fileNames)
#合并测试集
sourceFilePaths=[]
fileNames=pd.read_csv("../data/first_rank_index_20180307.csv")
for i in list(fileNames.iloc[:,0]):
    sourceFilePaths.append("../data/first_rank_data_20180307/"+str(i)+".txt")
dpp.mergeData(sourceFilePaths=sourceFilePaths,saveDir="../data",batchSize=10000,saveFileBaseName="test_",labels=None)
# In[] 构建降维学习器
sourceFilePaths=[]
for i in range(80):#剩下的训练集用于本地测试
    sourceFilePaths.append("../data/train_"+str(i)+".csv")
ipca=dpp.IPCA(filePaths=sourceFilePaths,k=150)#降到150维
joblib.dump(ipca,"ipca_150.model")#保存降维学习器
# In[] 使用降维学习器对训练集和测试集降维
#对训练集降维
sourceFilePaths=[]
for i in range(80):#剩下的训练集用于本地测试
    sourceFilePaths.append("../data/train_"+str(i)+".csv")
dpp.IPCATranform(ipca=ipca,filePaths=sourceFilePaths,saveFilePath="../data/train_IPCA.csv",useLastCol=False)
#对测试集降维
sourceFilePaths=[]
for i in range(10):
    sourceFilePaths.append("../data/test_"+str(i)+".csv")
dpp.IPCATranform(ipca=ipca,filePaths=sourceFilePaths,saveFilePath="../data/test_IPCA.csv",useLastCol=True)
# In[] 训练学习器xgbclassifier
#先随机欠采样
trainingSet=pd.read_csv("../data/train_IPCA.csv")
xgbs=Ensemble.Ensemble()
#开始训练
xgbs.fit(train=trainingSet)
#保存训练模型到本地
joblib.dump(xgbs,"XGBClassifier.model")
# In[] 开始预测结果
test=pd.read_csv("../data/test_IPCA.csv")
y=xgbs.predict(np.array(test))
res=pd.read_csv("../data/first_rank_index_20180307.csv")
res["type"]=y
res.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

































































