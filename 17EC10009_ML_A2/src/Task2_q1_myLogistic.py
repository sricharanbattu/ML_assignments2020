# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:53:32 2020

@author: lenovo
"""
import numpy as np

def cost_function(thetas,features,labels):
    """ 
    Returns the cost incurred due to a given theta,
    given theta and the training set"""
    m=features.shape[0]
    hthetas=1./(1+np.exp(-1*np.dot(features,thetas)))
    log_hthetas=np.log(hthetas)
    log_one_minus_hthetas=np.log(1-hthetas)
    cost=-1*(np.dot(labels,log_hthetas)+np.dot(1-labels,log_one_minus_hthetas))/m
    
    return cost

def update_GD(alpha,thetas,features,labels):
    """     
    This function returns the new theta given old theta ,the training data
    and the learning rate alpha
    """
    n_rows=features.shape[0]
    thetas_transpose_X=np.dot(features,thetas)
    h_thetas_X_inv=1+np.exp(-1*thetas_transpose_X)
    h_thetas_X=1./h_thetas_X_inv
    h_thetas_X_minus_Y=h_thetas_X-labels
    term_inside_sigma=np.dot(h_thetas_X_minus_Y,features)*alpha/n_rows
    
    return thetas-term_inside_sigma



def generate_probs(thetas,features):
    """ 
    This function gives the probability of the instances of training data
    belonging to class 1
    """
    probs= 1./(1+np.exp(-1*np.dot(features,thetas)))
    return probs
   
class LogisticRegressor():
    def __init__(self,alpha,features_train,labels_train,termination=1/(10**10)):
        self.alpha=alpha
        self.features_train=features_train
        self.labels_train=labels_train
        self.termination=termination
    def get_parameters(self):
        n_cols=self.features_train.shape[1]
        thetas=np.zeros(n_cols)                                                   #initialising the thetas/coefficients to zeros
        thetas_sqr_dffrnce=100                                                        # This denotes the difference between successive thetas in iterations of GD
        while(thetas_sqr_dffrnce>self.termination):                                          #Termination condition for gradient descent is difference between successive thetas is very small
            thetas_new=update_GD(self.alpha,thetas,self.features_train,self.labels_train)        
            thetas_sqr_dffrnce=np.sum(np.dot(thetas-thetas_new,thetas-thetas_new))
            thetas=thetas_new
        return thetas
    def get_probs(self,features):
        probs=generate_probs(self.get_parameters(),features)
        return probs