# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 01:29:05 2020

@author: lenovo
"""
import Task2_functions as p2f
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
print("\nSklearn's Logistic Regression started:Please wait until the metrics are printed\n")

"""                 DATA GENERATION   FROM THE CSV FILE INTO A NUMPY ARRAY          """
data1=np.genfromtxt('../data/datasetA.csv',delimiter=',')
data1=data1[1:,1:]
data1[:,0]=1

np.random.shuffle(data1)
n_rows=data1.shape[0]
n_cols=data1.shape[1]

"""                 EVALUATION METRIC OBJECTS INSTANTIATION                         """
rotate_order=[[0,1],[1,2],[2,3],[0,1],[1,2]]

train_accuracy=np.zeros((9,3))
test_accuracy=np.zeros((9,3))
train_precision=np.zeros((9,3))
test_precision=np.zeros((9,3))
train_recall=np.zeros((9,3))
test_recall=np.zeros((9,3))
confusion_for_test=np.zeros((9,5))
confusion_for_train=np.zeros((9,5))

mean_train_accuracy=np.zeros(9)                 #to store the mean of all evaluation metrics  ,generated from 3 fold cross validation
mean_test_accuracy=np.zeros(9)
mean_train_precision=np.zeros(9)
mean_test_precision=np.zeros(9)
mean_train_recall=np.zeros(9)
mean_test_recall=np.zeros(9)

for i in range(0,3):
    """                TRAIN AND TEST DATA GENERATION FOR CROSS  VALIDATION         """
    train_sub1          =      data1[int(rotate_order[i][0]*n_rows/3):int(rotate_order[i][1]*n_rows/3),:]
    train_sub2          =      data1[int(rotate_order[i+1][0]*n_rows/3):int(rotate_order[i+1][1]*n_rows/3),:]
    test                =      data1[int(rotate_order[i+2][0]*n_rows/3):int(rotate_order[i+2][1]*n_rows/3),:]
    train               =     np.concatenate((train_sub1,train_sub2))
    features_train      =     train[:,:-1]
    labels_train        =     train[:,-1]
    features_test       =     test[:,:-1]
    labels_test         =     test[:,-1]
    
    """                 TRAINING USING SKLEARN'S INBUILT LOGISTIC REGRESSOR          """ 
    clf                 =     LogisticRegression(random_state=0,solver="saga").fit(features_train, labels_train)
    prob_train          =     clf.predict_proba(features_train)
    prob_test           =     clf.predict_proba(features_test)
    
    """         STORING THE METRICS FOR EVALUATION OF THE TRAIN AND TEST DATA       """
    
    confusion_for_train =   p2f.generate_confusion(prob_train[:,1],labels_train)
    confusion_for_test  =   p2f.generate_confusion(prob_test[:,1],labels_test)
    train_accuracy[:,i]   =(confusion_for_train[:,0]+confusion_for_train[:,1])/confusion_for_train[:,4]
    test_accuracy[:,i]    =(confusion_for_test[:,0]+confusion_for_test[:,1])/confusion_for_test[:,4]
    train_precision[:,i]  =confusion_for_train[:,0]/(confusion_for_train[:,0]+confusion_for_train[:,2])
    test_precision[:,i]   =confusion_for_test[:,0]/(confusion_for_test[:,0]+confusion_for_test[:,2])
    train_recall[:,i]     =confusion_for_train[:,0]/(confusion_for_train[:,0]+confusion_for_train[:,3])
    test_recall[:,i]      =confusion_for_test[:,0]/(confusion_for_test[:,0]+confusion_for_test[:,3])


print("_"*120)
"""                         MEAN EVALUATION METRICS                             """
for i in range(0,9):                                                             
    mean_train_accuracy[i]=(train_accuracy[i,0]+train_accuracy[i,1]+train_accuracy[i,2])/3
    mean_test_accuracy[i]=(test_accuracy[i,0]+test_accuracy[i,1]+test_accuracy[i,2])/3
    mean_train_precision[i]=(train_precision[i,0]+train_precision[i,1]+train_precision[i,2])/3
    mean_test_precision[i]=(test_precision[i,0]+test_precision[i,1]+test_precision[i,2])/3
    mean_train_recall[i]=(train_recall[i,0]+train_recall[i,1]+train_recall[i,2])/3
    mean_test_recall[i]=(test_recall[i,0]+test_recall[i,1]+test_recall[i,2])/3

"""                         PRINTING THE METRICS                                    """

print("->Tst dentes test/validate")
print("->Acc denotes mean accuracy::Prscn denotes mean Precision::Rcl denotes mean Recall ")
print("->nan denotes that no 1's are predicted")
print("_"*120)
print("Threshold for 1","\t","Tst.Acc","\t","Tst.Prscn","\t","Tst.Rcl")
print("_"*120)
for i in range(0,9):
    print("{0:0.1f}  \t\t\t{1:0.4f}\t\t{2:0.4f}\t\t{3:0.4f}".format((i+1)/10,mean_test_accuracy[i],mean_test_precision[i],mean_test_recall[i]))
    
print("_"*120)