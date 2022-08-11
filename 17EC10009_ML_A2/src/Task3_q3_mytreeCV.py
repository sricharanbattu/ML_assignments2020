# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:58:51 2020

@author: lenovo
DECISON TREES
"""
import Task3_functions as T3f
import Task3_q1_myDecisionTree as T3q1
import pandas as pd
import numpy as np

print("\nDecision Tree has started::Please wait until the metrics are printed")

"""                             LOADING THE DATA SET                                """

data=pd.read_csv('../data/datasetB.csv')
columns=list(data.columns)
feature_columns=columns[1:-1]
target_column=columns[len(columns)-1]
n_rows=data.shape[0]

no_of_folds=3
rotate_order=[[0,1],[1,2],[2,3],[0,1],[1,2]]                                        #The order of instances to be considered as train or test in cross validation

"""                 EVALUATION METRIC OBJECTS INSTANTIATION                         """

train_accuracy=np.zeros(no_of_folds)                                                #This array stores the accuracy of the predictions in train data,for each cross validation  
test_accuracy=np.zeros(no_of_folds)                                                 #Similar to above ,but the data is test data
train_macro_precision=np.zeros(no_of_folds)                                         #Similar to above,but the metric is macro precision ,and the data is train data
test_macro_precision=np.zeros(no_of_folds)                                          #Similar to above:Metric stored is macro precision and data is test data
train_macro_recall=np.zeros(no_of_folds)                                            #Similar toabove:Metric stored is macro recall and data is train data
test_macro_recall=np.zeros(no_of_folds)                                             #Similar to above: Metric stored is macro recall and data is test data

"""                          CROSS VALIDATION STARTED                              """
for i in range(0,no_of_folds):
    """                TRAIN AND TEST DATA GENERATION FOR CROSS  VALIDATION         """
    trainsub1=data.iloc[int(rotate_order[i][0]*n_rows/3):int(rotate_order[i][1]*n_rows/3),:]
    trainsub2=data.iloc[int(rotate_order[i+1][0]*n_rows/3):int(rotate_order[i+1][1]*n_rows/3),:]
    test= data.iloc[int(rotate_order[i+2][0]*n_rows/3):int(rotate_order[i+2][1]*n_rows/3),:]
    train=pd.concat([trainsub1,trainsub2])
    train_labels=np.array(train[target_column])
    train_size=train_labels.shape[0]
    test_labels=np.array(test[target_column])
    test_size=test_labels.shape[0]
    train_predictions=np.zeros(train_size)                                             #For storing the predictions given by the model for train set
    test_predictions=np.zeros(test_size)                                               #For storing the predictions given by the model for test set
    """                               BUILDING THE MODEL                                 """
    Dtree=T3q1.DecisionTree(train,feature_columns,target_column)
    """                             PREDICTIONS USING THE TRAINED TREE                   """
    for j in range(0,train_size):
        train_predictions[j]=Dtree.predict(train.iloc[j,:])
    for j in range(0,test_size):
        test_predictions[j]=Dtree.predict(test.iloc[j,:])
    
    """                             EVALUATING THE MODEL USING THE PREDICTIONS           """
    train_accuracy[i]=T3f.macro_metrics(train_predictions,train_labels)[0]
    test_accuracy[i]=T3f.macro_metrics(test_predictions,test_labels)[0]
    train_macro_precision[i]=T3f.macro_metrics(train_predictions,train_labels)[1]
    test_macro_precision[i]=T3f.macro_metrics(test_predictions,test_labels)[1]
    train_macro_recall[i]=T3f.macro_metrics(train_predictions,train_labels)[2]
    test_macro_recall[i]=T3f.macro_metrics(test_predictions,test_labels)[2]


"""                                      CROSS VALIATION OVER                             """

"""                                      MEAN  MACRO METRICS                               """

train_mean_accuracy=np.mean(train_accuracy)
test_mean_accuracy=np.mean(test_accuracy)
train_mean_precision=np.mean(train_macro_precision)
test_mean_precision=np.mean(test_macro_precision)
train_mean_recall=np.mean(train_macro_recall)
test_mean_recall=np.mean(test_macro_recall)

"""                             PRINT MEAN MACRO METRICS                            """
#print("_"*120)
#print("\t\t\t\tMean Evaluation Metrics For Training")
#print(" Metrics\t\tTrain Mean Accuracy\tTrain Mean Macro Precision\tTrain Mean Macro Recall")
#print("_"*120)
#print(" Values \t\t{0:0.4f}\t\t\t{1:0.4f}\t\t\t{2:0.4f}".format(train_mean_accuracy,train_mean_precision,train_mean_recall))
#print("_"*120)
print('\n')
print("->Valid. denotes Validation")
print("_"*120)
print("\t\t\t\tMean Evaluation Metrics For Testing Using Cross Validation")
print("_"*120)
print(" Metrics\t\tValid. Mean Accuracy\tValid. Mean Macro Precision\tValid. Mean Macro Recall")
print("_"*120)
print(" Values \t\t{0:0.4f}\t\t\t{1:0.4f}\t\t\t\t{2:0.4f}".format(test_mean_accuracy,test_mean_precision,test_mean_recall))
print("_"*120)
print('\n')

