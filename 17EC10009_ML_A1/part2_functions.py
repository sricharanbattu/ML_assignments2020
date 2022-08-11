# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:30:20 2020

@author: sricharanbattu
"""
import poly_regression as pr

import numpy as np
import matplotlib.pyplot as plt


def plot_line_curve(B,w,labels,deg):
    """
    This function plots the true data points and
    the fitted curve given by the features present 
    in B,by weighting with w
    
    """
    m=np.amin(B[:,1])
    M=np.amax(B[:,1])
    row=np.shape(B)[1]
    plt.style.use("ggplot")
    x=np.array(np.linspace(m,M,1000))
    A=pr.generate_features(x,row-1)
    y=np.dot(A,w)
    fig=plt.figure(figsize=(10,10))
    plt.style.use("ggplot")
    ax=fig.add_subplot(1,1,1)
    ax.scatter(B[:,1],labels,color='blue',s=7,label='True  data')
    ax.plot(x,y,color='red',label='Polynomial fit')
    ax.set_xlabel("Features")
    ax.set_ylabel("Values")
    ax.set_title(f"Polynomial fit of degree{deg}")
    plt.legend(loc='best')
    plt.savefig(f'q2a_fitdegree_{deg}.jpeg')
    plt.show()
    return

def plot_error(train_error_container,test_error_container,n):
    """
    This function plots the train and test errors as a function
    of the degree of the polynomial used in Regression.train_error_container
    and test_error_container are one dimensional arrays containing errors of 
    each model.
    """
    x=np.array(range(1,n+1))
    plt.style.use("ggplot")
    fig= plt.figure(figsize=(10,10))
    ax=fig.add_subplot(1,1,1)
    ax.plot(x,train_error_container,color='red',label="Train error")
    ax.plot(x,test_error_container,color='blue',label="Test error")
    ax.set_xlabel('Degree of polynomial fit')
    ax.set_ylabel('Error')
    ax.set_title(f"degree vs error ")
    plt.legend(loc='best')
    plt.savefig('q2b_errors.jpeg')
    plt.show()
    return