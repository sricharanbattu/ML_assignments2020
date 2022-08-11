# -*- coding: utf-8 -*-
"""
Spyder Editor

NAME      : BATTU SRI CHARAN
ROLL NO   : 17EC10009
gmail id  :sricharanbattu@gmail.com
"""

import poly_regression as pr

import numpy as np
import matplotlib.pyplot as plt
import sys



sysargc=len(sys.argv)
if sysargc>=2:
    thr=int(sys.argv[1])#threshold is gradient step terminator
else:
    thr=8

if sysargc>=4:
    degree=int(sys.argv[3])#degree is the max degree of the polynomial wished to be observed
else:
    degree=9

if sysargc>=3:
    alpha=float(sys.argv[2])#alpha is the learning rate
else:
    alpha=0.05

threshold=1/(10**thr)


train=pr.load_data("train.csv")
test=pr.load_data("test.csv")
features_train=train[:,0]
labels_train=train[:,1]
features_test=test[:,0]
labels_test=test[:,1]
if __name__=="__main__":#For plotting of train and test data
    pr.plot_file(features_train,labels_train,"train_set")
    pr.plot_file(features_test,labels_test,"test_Set")

labels_pred_train=np.zeros(( np.shape(train)[0],degree))#contains labels predicted by each polynomial model for train/test
labels_pred_test=np.zeros((np.shape(test)[0],degree))

error_test=np.zeros(degree)#For storing the test errors for each degree
error_train=np.zeros(degree)#For storing the train errors for each degree
weight=np.zeros((degree,degree+1))#nth row of weights stores the weights of nth deg polynomial,kth column stores kth degree coefficient



if __name__=="__main__":#printing the hyper parameters
    
    print("_"*120)
    print("Learning rate        :\t",alpha)
    print("Termination limit    :\t",threshold)
    print("_"*120)    
    print("\tPolynomial degree\t\t\tTrain error\t\t\tTest error\t\t\t\t\t")
    print("_"*120)
    
    
B_train_big=pr.generate_features(features_train,degree)
B_test_big=pr.generate_features(features_test,degree)


B_train_big=pr.generate_features(features_train,degree)
B_test_big=pr.generate_features(features_test,degree)
#Minimise the squared Loss Using Gradient descent
for deg in range(1,degree+1):
    B=B_train_big[:,:deg+1]
    B_test=B_test_big[:,:deg+1]
    w=np.zeros(deg+1)
    termination=100
    while(termination>threshold):
        
        w_updated=pr.update(w,B,labels_train,alpha)
        termination=np.sum(np.square(w-w_updated))
        w=w_updated
    
    labels_pred_train[:,deg-1]=np.dot(B,w_updated)#Filling the labels predicted
    labels_pred_test[:,deg-1]=np.dot(B_test,w_updated)
    error_test[deg-1]=pr.mean_squared_error(labels_pred_test[:,deg-1],labels_test)#storing the errors
    error_train[deg-1]=pr.mean_squared_error(labels_pred_train[:,deg-1],labels_train)
    for temp in range(0,deg+1):
        weight[deg-1,temp]=w_updated[temp]#storing the weights
        
    #Print the results
    if __name__=="__main__":
        
        print("     \t\t{0:}\t\t\t\t{1:0.4f}\t\t\t\t{2:0.4f}\t\t\t\t\t\n".format(deg,error_train[deg-1],error_test[deg-1]))#printing the errors
    
 
        
print("_"*120)
if __name__=="__main__":
    if(degree<=9):
        pr.print_weights(weight,list(range(0,9)))#printing the weights obtained from gradient descent for each polynomial
    
  
    




