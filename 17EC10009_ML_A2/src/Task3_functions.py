# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:52:07 2020

@author: lenovo
"""
import numpy as np

def accuracy(preds,actual):
    """
    Returns the proportion of  no of instances correctly predicted 
    """
    return np.sum(preds==actual)/preds.shape[0]

def precision(preds,actual):
    """
    Returns a 1xn array whose values are precisions corresponding to each class
    given the predicted values and actual values.n denotes no of classes
    """
    unique_vals=np.unique(preds)
    n=unique_vals.shape[0]
    pre=np.zeros(n)
    for i in range(0,n):
        label=unique_vals[i]
        x=preds==label
        y=actual==label
        pre[i]=np.dot(x+0,y+0)/np.sum(x)
        
    return pre

def recall(preds,actual):
    """
    Returns a 1xn array whose values are recall corresponding to each class
    given the predicted values and actual values.n denotes no of classes
    
    """
    unique_vals=np.unique(preds)
    n=unique_vals.shape[0]
    rcl=np.zeros(n)
    for i in range(0,n):
        label=unique_vals[i]
        x=preds==label
        y=actual==label
        rcl[i]=np.dot(x+0,y+0)/np.sum(y)
    return rcl

def macro_metrics(preds,actual):
    """
    Returns a 1x3 array containing the averages of accuracy,precision and recall 
    corresponding to each class given predictions and true label classes
    """
    unique_vals=np.unique(actual)
    n=unique_vals.shape[0]
    acc=accuracy(preds,actual)
    pre=precision(preds,actual)
    rcl=recall(preds,actual)
    return [acc,np.mean(pre),np.mean(rcl)]


