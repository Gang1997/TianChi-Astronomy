# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:42:41 2018

@author: Administrator
"""
from copy import deepcopy
import pandas as pd
import codecs
from sklearn.decomposition import IncrementalPCA
#de.
def mergeData(sourceFilePaths,saveDir,batchSize,saveFileBaseName,labels=None):
    """将原始txt数据分批保存成csv文件""" 
    strBuffer,count,fileID=[],0,0
    for i in sourceFilePaths:
        strTxt=codecs.open(i).read() 
#        if(labels is not None):
#            strTxt+=(','+labels[labels["id"]==int(i.split("/")[-1].split(".")[0])].iloc[0,1])            
        strBuffer.append(strTxt.split(','))
        count+=1
        if(count==batchSize):
#            print(fileID)##########################
            df=pd.DataFrame(strBuffer) 
            if(labels is not None):
                df["type"]=list(labels.iloc[list(range(fileID*batchSize,(fileID+1)*batchSize)),-1])
            df.to_csv(saveDir+"/"+saveFileBaseName+str(fileID)+".csv",index=False)
            strBuffer,df,count=[],None,0
            fileID+=1
    if(df is not None):
        df=pd.DataFrame(strBuffer) 
        if(labels is not None):
            df["type"]=list(labels.iloc[list(range(fileID*batchSize,len(labels))),-1])
        df.to_csv(saveDir+"/"+saveFileBaseName+str(fileID)+".csv",index=False)
        df=None
    
def IPCA(filePaths,k):
    """增量式降维,返回降维器"""
    ipca=IncrementalPCA(n_components=k)
    for i in range(len(filePaths)):#对于每个文件
#        print("降维学习",filePaths[i]) 
        df=pd.read_csv(filePaths[i]).iloc[:,0:-1]
        ipca.partial_fit(df)
    return ipca
def IPCATranform(ipca,filePaths,saveFilePath,useLastCol):
    """使用降维器进行降维"""
    dfs=None
    df=pd.read_csv(filePaths[0])  
    if(not useLastCol):
        y=deepcopy(df.iloc[:,-1])
        dfs=pd.DataFrame(ipca.transform(df.iloc[:,0:-1]))
        dfs["type"]=y
        df=None
    else:
        dfs=pd.DataFrame(ipca.transform(df))
        
    for i in range(1,len(filePaths)):
        print("降维",i)
        df=pd.read_csv(filePaths[i])    
        if(not useLastCol):
            y=deepcopy(df.iloc[:,-1])
            df=pd.DataFrame( ipca.transform( df.iloc[:,0:-1]   )    )
            df["type"]=y
            dfs=dfs.append(df,ignore_index=True)
            df=None
        else:
            dfs=dfs.append(pd.DataFrame(ipca.transform(df)),ignore_index=True)
    dfs.to_csv(saveFilePath,index=False)
    
    
    
    
    
    
    
    