# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:04:56 2020

@author: lenovo
"""
import Task2_functions as p2f
import Task2_q1_myLogistic as T2q1

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

print("\nLogistic Regression has started:Please wait until the metrics are printed")

"""                             HYPER PARAMETERS                                    """
alpha=0.1                                           #learning rate
thr=8
termination=1/(10**thr)                             #provides the termination condition for gradient descent

"""                 DATA GENERATION   FROM THE CSV FILE INTO A NUMPY ARRAY          """

data1=np.genfromtxt('../data/datasetA.csv',delimiter=',')
data1=data1[1:,:]
data1[:,0]=1                                        #adding ones for calculations of the constant in the coefficients


#np.random.shuffle(data1)                           #shuffling the data to eliminate any biases in ordering of the data
n_rows=data1.shape[0]                               #no of instances 
n_cols=data1.shape[1]                               #no of features that includes the appended column

"""                 EVALUATION METRIC OBJECTS INSTANTIATION                         """
rotate_order=[[0,1],[1,2],[2,3],[0,1],[1,2]]    #for easing the extraction of train and test data in 3 fold cross validation

train_accuracy=np.zeros((9,3))                  #to store the accuracy of train data as a function of probability threshold in 3 folds
test_accuracy=np.zeros((9,3))                   #above step for test data accuracy
train_precision=np.zeros((9,3))                 #above step for train data precision
test_precision=np.zeros((9,3))                  #above step for test data precision
train_recall=np.zeros((9,3))                    #above step for train data recall
test_recall=np.zeros((9,3))                     #above step for test data recall
confusion_for_test=np.zeros((9,5))              # to store the no of true positives,true negatives,false positives,false negatives and no of  instances of train data as a function of prob threshold
confusion_for_train=np.zeros((9,5))             # above step for test data

mean_train_accuracy=np.zeros(9)                 #to store the mean of all evaluation metrics 
mean_test_accuracy=np.zeros(9)
mean_train_precision=np.zeros(9)
mean_test_precision=np.zeros(9)
mean_train_recall=np.zeros(9)
mean_test_recall=np.zeros(9)
"""                          CROSS VALIDATION STARTED                              """
for i in range(0,3):
    """                TRAIN AND TEST DATA GENERATION FOR CROSS  VALIDATION         """
    train_sub1=data1[int(rotate_order[i][0]*n_rows/3):int(rotate_order[i][1]*n_rows/3),:]
    train_sub2=data1[int(rotate_order[i+1][0]*n_rows/3):int(rotate_order[i+1][1]*n_rows/3),:]
    test=      data1[int(rotate_order[i+2][0]*n_rows/3):int(rotate_order[i+2][1]*n_rows/3),:]
    train=     np.concatenate((train_sub1,train_sub2))
    features_train=train[:,:-1]
    labels_train  =train[:,-1]
    features_test =test[:,:-1]
    labels_test   =test[:,-1]
    """                     TRAINING ON TRAIN DATA USING GRADIENT DESCENT           """
    
    clf=T2q1.LogisticRegressor(alpha,features_train,labels_train,termination)
    thetas=clf.get_parameters()

    """         STORING THE METRICS FOR EVALUATION OF THE TRAIN AND TEST DATA       """
    probs_for_train       =clf.get_probs(features_train)                    #probabilities that an instance belongs to class 1 for the train set
    probs_for_test        =clf.get_probs(features_test)                     #the above step for test set
    confusion_for_train   =p2f.generate_confusion(probs_for_train,labels_train)
    confusion_for_test    =p2f.generate_confusion(probs_for_test,labels_test)
    train_accuracy[:,i]   =(confusion_for_train[:,0]+confusion_for_train[:,1])/confusion_for_train[:,4]
    test_accuracy[:,i]    =(confusion_for_test[:,0]+confusion_for_test[:,1])/confusion_for_test[:,4]
    train_precision[:,i]  =confusion_for_train[:,0]/(confusion_for_train[:,0]+confusion_for_train[:,2])
    test_precision[:,i]   =confusion_for_test[:,0]/(confusion_for_test[:,0]+confusion_for_test[:,2])
    train_recall[:,i]     =confusion_for_train[:,0]/(confusion_for_train[:,0]+confusion_for_train[:,3])
    test_recall[:,i]      =confusion_for_test[:,0]/(confusion_for_test[:,0]+confusion_for_test[:,3])
    print(f" Training and Testing completed for  fold set: {i}")                            #indication for successful completition of cross validation for the ith fold 
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
