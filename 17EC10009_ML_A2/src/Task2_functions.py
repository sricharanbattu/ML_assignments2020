# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:45:13 2020

@author: lenovo
"""
import numpy as np
def generate_confusion(probs,labels):
    """
    generates the confusion matrix for each threshold of 
    an instance belonging to class 1, given the probabilities 
    belonging to class 1 and the true labels
    """
    A=np.zeros((9,5))
    for i in range(0,9):
        A[i,:]=generate_confusion_terms(probs,labels,(i+1)/10)
    
    return A 


def generate_confusion_terms(probs,labels,thr):
    """
    This function generates the confusion matrix for a given threshold 
    of instances to be classified as  class 1,true labels of the instances, and 
    probabilities of the instances to belong to class 1.
    The confusion matrix is spread out as a 1 dimensional array containing True positives,
    True negatives,false positives,false negatives,total no of instances
    """
    n_rows=labels.shape[0]
    labels_pred=np.zeros(n_rows)
    labels_pred[probs>thr]=1
    labels_pred[probs<=thr]=0
    
    TP=np.dot(labels_pred,labels)
    TN=np.dot(1-labels_pred,1-labels)
    FP=np.dot(labels_pred,1-labels)
    FN=np.dot(1-labels_pred,labels)
    
    return np.array([TP,TN,FP,FN,TP+TN+FP+FN])

    