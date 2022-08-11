# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:39:39 2020

@author: lenovo
"""
"""
This file is my solution to the evaluation metric Normalised Mutual Information
NMI. 
"""

import pandas as pd
import numpy as np



def entropy(label_counts):
    """
    This function takes as arguments:
        1. label_counts : A list whose elements are number of instances of a 
        particular cluster.
    It calculates the entropy of a distribution
    """
    
    l=len(label_counts)
    label_c=np.array(label_counts)              
    s=np.sum(label_c)                                                           #'s' stores the total no of values 
    h=0                                                                         # 'h' means entropy of the distribution
    for i in range(0,l):                                                        # This loop finds the entropy 
        x=label_c[i]/s
        h=h-(x*np.log2(x))
    return h

def conditional_entropy(cluster_list,dataframe):
    """
    This function takes as arguments:
        1. cluster_list : The clusters found by the algorithm
        2. dataframe    : The original clusters
    This function finds the conditional entropy H(Y/C=c) and finds the weighted 
    sum of them, which gives the total conditional entropy
    """
    l=len(cluster_list)
    total_counts=dataframe.shape[0]
    mi=0
    for i in range(0,l):
        d=dataframe.iloc[cluster_list[i]]                                       #'d' stores the actual distribution of a particular predicted cluster 
        c_counts=d.shape[0]
        vc=d.value_counts()
        mi=mi+(entropy(vc)*c_counts/total_counts)                               # the weighted entropy of conditional distribution and the cumulative sum
    return mi

def NMI(cluster_list,dataframe):
    """
    This function takes as arguments :
        1. cluster_list: the list of clusters found by the algorithm
        2. dataframe : actual clusters
    this function outputs normalised mutual information: 2*MI/(H(Y)+H(C)),
    where MI=H(Y)-H(Y/C)
    """
    y_counts=dataframe.value_counts()                                           #y_counts denote the freq distribution of actual clusters
    h_y=entropy(y_counts)                                                       #h_y denotes H(Y),the entropy of actual cluster distribution
    l=len(cluster_list)
    c_counts=[]                                                                 #c_counts the freq distribution of predicted clusters
    for i in range(0,l):
        c_counts.append(len(cluster_list[i]))
    h_c=entropy(c_counts)                                                       #h_c denotes H(C) ,the entropy of the predicted cluster distribution
    mi=conditional_entropy(cluster_list,dataframe)                              #mi denotes conditional entropy  H(Y/C)
    MI=h_y-mi                                                                   #MI denotes mutual information bw Y and C
    nmi=2*MI/(h_y+h_c)                                                          #nmi denotes normalised mutual information
    return nmi
    