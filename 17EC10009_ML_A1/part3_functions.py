# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:06:28 2020

@author: sricharan batttu
NAME      : BATTU SRI CHARAN
ROLL NO   : 17EC10009
gmail id  :sricharanbattu@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

def update_ridge(weights,features,labels,alpha,lam):
    """
    This function implements the gradient descent updating step
    in Ridge regression
    """
    
    number_of_points=np.shape(features)[0]
    P=np.dot(features,weights)-labels
    return (weights-alpha*(1/number_of_points)*(np.dot(P,features)+lam*weights))



def update_lasso(weights,features,labels,alpha,lam):
    """
    This function implements the gradient descent updating step
    in Lasso Regression
    """
    truth =2*(weights>=0)-1
   
    truth[0]=0
    number_of_points=np.shape(features)[0]
    P=np.dot(features,weights)-labels
    return (weights-alpha*(1/number_of_points)*(np.dot(P,features)+lam*truth/2))



def LS_error(y_pred,y_true):
    """
    This function computes the error
    when train and test sets are given
    """
    size=np.shape(y_pred)[0]
    return np.sum(np.square(y_pred-y_true))/(2*size)

def plot_error_p3(train_error_ridge,test_error_ridge,train_error_lasso,test_error_lasso,lambdas,deg_array):
    plt.style.use('dark_background')
    fig1=plt.figure(figsize=(10,10))
    ax1=fig1.add_subplot(1,1,1)
    ax1.plot(lambdas,train_error_ridge[0,:],label=f'Ridge Train Error for degree {deg_array[0]} ',color='indianred',marker='o')
    ax1.plot(lambdas,test_error_ridge[0,:],label=f'Ridge Test Error for degree {deg_array[0]} ',color='royalblue',marker='o')
    ax1.plot(lambdas,train_error_lasso[0,:],label=f'Lasso Train Error for degree {deg_array[0]} ',color='indianred',linestyle='dashed',marker='o')
    ax1.plot(lambdas,test_error_lasso[0,:],label=f'Lasso Test Error for degree {deg_array[0]} ',color='royalblue',linestyle='dashed',marker='o')
    ax1.set_xlabel('Regularization parameter lambda')
    ax1.set_ylabel('Errors for min error model')
    ax1.set_title('Ridge and Lasso comparison for min error model')
    plt.legend(loc='best')
    plt.savefig('q3_minErrorModel.jpeg')
    plt.show()
    
    fig2=plt.figure(figsize=(10,10))
    ax2=fig2.add_subplot(1,1,1)
    ax2.plot(lambdas,train_error_ridge[1,:],label=f'Ridge Train Error for degree {deg_array[1]} ',color='indianred',marker='o')
    ax2.plot(lambdas,test_error_ridge[1,:],label=f'Ridge Test Error for degree {deg_array[1]} ',color='royalblue',marker='o')
    ax2.plot(lambdas,train_error_lasso[1,:],label=f'Lasso Train Error for degree {deg_array[1]} ',color='indianred',linestyle='dashed',marker='o')
    ax2.plot(lambdas,test_error_lasso[1,:],label=f'Lasso Test Error for degree {deg_array[1]} ',color='royalblue',linestyle='dashed',marker='o')
    ax2.set_xlabel('Regularization parameter lambda')
    ax2.set_ylabel('Errors for max error model')
    ax2.set_title('Ridge and Lasso comparison for max error model')
    plt.legend(loc='best')
    plt.savefig('q3_MaxErrorModel.jpeg')
    plt.show()
    return

def print_weights_p3(lambdas,weights,degrees,f=sys.stdout):
    icount=len(lambdas)
    jcount=len(degrees)
    print("Lambda \tPolynomial degree \tconstn \tcoefx1 \tcoefx2 \tcoefx3 \tcoefx4 \tcoefx5 \tcoefx6 \tcoefx7 \tcoefx8 \tcoefx9",file=f)
    
    print("_"*120,file=f)
    for i in range(0,icount):
        for j in range(0,jcount):
            print("{0:}  \t\t{1:}\t".format(lambdas[i],degrees[j]),end=' ',file=f)
            print("\t",end=' ',file=f)
            for k in range(0,degrees[j]+1):
                print("{0:0.4g}".format(weights[2*i+j,k]),end=' ',file=f)
            print("\n",file=f)
    print("_"*120,file=f)
    return

def print_errors_p3(lambdas,train_errors,test_errors,degrees,f=sys.stdout):
    """
    This function prints errors passed by train errors and test errors and
    print them systematically separating them into columns.
    lambdas is a list;train_errors and test_errors  is 2 dimensional array
    having twice the rows as the size of lambda;degrees is the array having
    minimum error and maximum error models.
    """
    icount=len(lambdas)
    jcount=len(degrees)
    print("\tlambda\t\t\tPolynomial degree\t\tTrain error\t\tTest error\t\t\t\t\t",file=f)
    for i in range(0,icount):
        for j in range(0,jcount):
            print("\t{0:}\t\t\t     \t{1:}\t\t\t{2:0.4f}\t\t\t{3:0.4f}\t\t\t\t\t".format(lambdas[i],degrees[j],train_errors[j,i],test_errors[j,i]),file=f)
    print("_"*120,file=f)
    return