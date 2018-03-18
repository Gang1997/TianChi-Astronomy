# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:11:34 2018

@author: Administrator
"""
import numpy as np
def PRF(TP,TruNum,PredRum):
    """ 计算Precision Recall 和F1"""
    precision,recall,FScore=0.0,0.0,0.0
    if(PredRum !=0):
        precision=TP/PredRum
    if(TruNum !=0):
        recall=TP/TruNum
    if((precision+recall)!=0):
        FScore=2.0*precision*recall/(precision+recall)
    return(precision,recall,FScore)
    
def eval_MF1(y_predicted=None,y_true=None):
    trueClass,predClass=list(y_true),list(y_predicted)
    
    qsoTP,qsoNum,qsoPredNum=0.0,0.0,0.0
    starTP,starNum,starPredNum=0.0,0.0,0.0
    unknownTP,unknownNum,unknownPredNum=0.0,0.0,0.0
    galaxyTP,galaxyNum,galaxyPredNum=0.0,0.0,0.0
    for tru,pred in zip(trueClass,predClass):
        if(pred=="star" and tru=="star"):
            starTP+=1
        if(tru=="star"):
            starNum+=1
        if(pred=="star"):
            starPredNum+=1
        ###########################
        if(pred=="galaxy" and tru=="galaxy"):
            galaxyTP+=1
        if(tru=="galaxy"):
            galaxyNum+=1
        if(pred=="galaxy"):
            galaxyPredNum+=1
         ###########################
        if(pred=="unknown" and tru=="unknown"):
            unknownTP+=1
        if(tru=="unknown"):
            unknownNum+=1
        if(pred=="unknown"):
            unknownPredNum+=1
        ###########################
        if(pred=="qso" and tru=="qso"):
            qsoTP+=1
        if(tru=="qso"):
            qsoNum+=1
        if(pred=="qso"):
            qsoPredNum+=1
    
    starP,starR,starF1=PRF(starTP,starNum,starPredNum)
    qsoP,qsoR,qsoF1=PRF(qsoTP,qsoNum,qsoPredNum)
    unknownP,unknownR,unknownF1=PRF(unknownTP,unknownNum,unknownPredNum)
    galaxyP,galaxyR,galaxyF1=PRF(galaxyTP,galaxyNum,galaxyPredNum)
    
    MF1=(starF1+qsoF1+unknownF1+galaxyF1)/4.0
    return ("MF1",-1*MF1)#为了目标函数
def eval_DiffSum(y_predicted=None,y_true=None):
    trueClass,predClass=list(y_true),list(y_predicted)
    
    qsoTP,qsoNum,qsoPredNum=0.0,0.0,0.0
    starTP,starNum,starPredNum=0.0,0.0,0.0
    unknownTP,unknownNum,unknownPredNum=0.0,0.0,0.0
    galaxyTP,galaxyNum,galaxyPredNum=0.0,0.0,0.0
    for tru,pred in zip(trueClass,predClass):
        if(pred=="star" and tru=="star"):
            starTP+=1
        if(tru=="star"):
            starNum+=1
        if(pred=="star"):
            starPredNum+=1
        ###########################
        if(pred=="galaxy" and tru=="galaxy"):
            galaxyTP+=1
        if(tru=="galaxy"):
            galaxyNum+=1
        if(pred=="galaxy"):
            galaxyPredNum+=1
         ###########################
        if(pred=="unknown" and tru=="unknown"):
            unknownTP+=1
        if(tru=="unknown"):
            unknownNum+=1
        if(pred=="unknown"):
            unknownPredNum+=1
        ###########################
        if(pred=="qso" and tru=="qso"):
            qsoTP+=1
        if(tru=="qso"):
            qsoNum+=1
        if(pred=="qso"):
            qsoPredNum+=1
    
    starP,starR,starF1=PRF(starTP,starNum,starPredNum)
    qsoP,qsoR,qsoF1=PRF(qsoTP,qsoNum,qsoPredNum)
    unknownP,unknownR,unknownF1=PRF(unknownTP,unknownNum,unknownPredNum)
    galaxyP,galaxyR,galaxyF1=PRF(galaxyTP,galaxyNum,galaxyPredNum)
    return ("DiffSum",np.abs(starP-starR)+np.abs(qsoP-qsoR)+np.abs(unknownP-unknownR)+np.abs(galaxyP-galaxyR))#为了目标函数
    

def MF(y_predicted=None,y_true=None):
    trueClass,predClass=list(y_true),list(y_predicted)
    
    qsoTP,qsoNum,qsoPredNum=0.0,0.0,0.0
    starTP,starNum,starPredNum=0.0,0.0,0.0
    unknownTP,unknownNum,unknownPredNum=0.0,0.0,0.0
    galaxyTP,galaxyNum,galaxyPredNum=0.0,0.0,0.0
    for tru,pred in zip(trueClass,predClass):
        if(pred=="star" and tru=="star"):
            starTP+=1
        if(tru=="star"):
            starNum+=1
        if(pred=="star"):
            starPredNum+=1
        ###########################
        if(pred=="galaxy" and tru=="galaxy"):
            galaxyTP+=1
        if(tru=="galaxy"):
            galaxyNum+=1
        if(pred=="galaxy"):
            galaxyPredNum+=1
         ###########################
        if(pred=="unknown" and tru=="unknown"):
            unknownTP+=1
        if(tru=="unknown"):
            unknownNum+=1
        if(pred=="unknown"):
            unknownPredNum+=1
        ###########################
        if(pred=="qso" and tru=="qso"):
            qsoTP+=1
        if(tru=="qso"):
            qsoNum+=1
        if(pred=="qso"):
            qsoPredNum+=1
    
    starP,starR,starF1=PRF(starTP,starNum,starPredNum)
    qsoP,qsoR,qsoF1=PRF(qsoTP,qsoNum,qsoPredNum)
    unknownP,unknownR,unknownF1=PRF(unknownTP,unknownNum,unknownPredNum)
    galaxyP,galaxyR,galaxyF1=PRF(galaxyTP,galaxyNum,galaxyPredNum)
    
    MF1=(starF1+qsoF1+unknownF1+galaxyF1)/4.0
    DiffSum=np.abs(starP-starR)+np.abs(qsoP-qsoR)+np.abs(unknownP-unknownR)+np.abs(galaxyP-galaxyR)
    return (MF1,DiffSum,{"star":(starTP,starNum,starPredNum,starR,starP), 
                 "galaxy":(galaxyTP,galaxyNum,galaxyPredNum,galaxyR,galaxyP), 
                 "qso":(qsoTP,qsoNum,qsoPredNum,qsoR,qsoP),  
                 "unknown":(unknownTP,unknownNum,unknownPredNum,unknownR,unknownP)})






















