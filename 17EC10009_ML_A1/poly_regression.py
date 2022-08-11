# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:39:30 2020

@author: sricharan battu
NAME      : BATTU SRI CHARAN
ROLL NO   : 17EC10009
gmail id  :sricharanbattu@gmail.com
"""
#THIS FILE WRITES THE FUNCTIONS RELATING TO THE POLYNOMIAL REGRESSION 
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(filename):
    """
    This function reads the data from the 
    given csv file.The argument ideally should have only 
    two features which are numerical. This discards the 
    names of the columns and return only the data points
    """
    A=np.genfromtxt(filename,delimiter=',');
    return A[1:,]

def plot_file(features,labels,name):
    """
    This functions plots a scatter plot 
    taking features as x and labels as y
    """
    if name=='train_set':
        c='red'
    else:
        c='blue'
    plt.style.use('ggplot')
    fig= plt.figure(figsize=(10,10))
    ax=fig.add_subplot(1,1,1)
    ax.scatter(features,labels,color=c,s=7)
    ax.plot([np.min(features),np.max(features)+0.1],[0,0],color='black')
    ax.plot([0,0],[np.min(labels)-0.1,np.max(labels)+0.1],color='black')
    ax.set_xlabel("Feature value of "+name)
    ax.set_ylabel("Output values of "+name)
    ax.set_xlim(np.min(features)-0.1,np.max(features)+0.1)
    ax.set_ylim(np.min(labels)-0.1,np.max(labels)+0.1)
    ax.set_title('Plot of features vs output values of '+ name)
    
    #plt.show()
    plt.savefig(f'q1a_{name}.jpeg')
    plt.show()
    return
    

def generate_features(features,n):
    """
    A must be a column vector
    This function generates the polynomial features of A upto power n
    and returns the corresponding array
    """
    rows=np.shape(features)
    B=np.zeros((rows[0],n+1))
    B[:,0]=np.ones(rows[0])
    for i in range(1,n+1):
        B[:,i]=np.power(features,i)
    return B

def Loss(B,weights,labelspace):
    """
    This function returns the squared loss when the 
    feature space,the corresponding coefficients and the 
    label sapce are given
    
    """
    
    return np.sum(np.square(np.dot(B,weights) - labelspace))/np.shape(B)[0]

def update(w,B,labels,alpha):
    """
    Thisfunction implements updation step in 
    gradient descent algorithm for regression
    The updation step can be performed using 
    Matrix multiplication
    
    """
    B_shape=np.shape(B)
    row=B_shape[0]
    col=B_shape[1]
    P=np.dot(B,w)-labels
    return w-alpha*(1/row)*(np.dot(P,B))

def mean_squared_error(y_pred,y_actual):
    """
     This function returns mean squared error between
     y_pred and y_test
     
    """
    return np.sum(np.square(y_pred-y_actual))/(2*np.shape(y_actual)[0])

def print_weights(weight,degrees,f=sys.stdout):
    print("_"*120,file=f)
    print("Polynomial degree\tconst \tcoef_x \tcoefx2 \tcoefx3 \tcoefx4 \tcoefx5 \tcoefx6 \tcoefx7 \tcoefx8 \tcoefx9",file=f)
    print("_"*120,file=f)
    for i in degrees:
        print("\t",i+1,end='\t\t',file=f)
        for j in range(0,i+2):
            print("{0:0.4f}".format(weight[i,j]),end='\t',file=f)
        print("\n",file=f)
    print("_"*120)

def print_errors(error_test,error_train,deglist,f=sys.stdout):
    print("_"*120,file=f)
    print("\tPolynomial degree\t\t\tTrain error\t\t\tTest error\t\t\t\t\t",file=f)
    print("_"*120,file=f)
    for deg in deglist:
        print("     \t\t{0:}\t\t\t\t{1:0.4f}\t\t\t\t{2:0.4f}\t\t\t\t\t\n".format(deg,error_train[deg-1],error_test[deg-1]),file=f)

    print("_"*120,file=f)
    return
