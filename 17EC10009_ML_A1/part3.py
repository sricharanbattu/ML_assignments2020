# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:28:26 2020

@author: sricharan battu
NAME      : BATTU SRI CHARAN
ROLL NO   : 17EC10009
gmail id  :sricharanbattu@gmail.com
"""
import main_file as mainf

import poly_regression as pr
import part3_functions as p3f

import numpy as np
import matplotlib.pyplot as plt

###rIDGE REGRESSION
trainerror_polyreg=mainf.error_train
testerror_polyreg=mainf.error_test
weight_polyreg=mainf.weight
m=np.argmin(trainerror_polyreg)#degree of min error model -1
M=np.argmax(trainerror_polyreg)#degree of max error model -1
highest_degree=mainf.degree
alpha=mainf.alpha
threshold=mainf.threshold
lambdas=[0.25,0.50,0.75,1]
lambda_size=len(lambdas)

train_data=pr.load_data("train.csv")
test_data=pr.load_data("test.csv")
train_features=train_data[:,0]
train_labels=train_data[:,1]
test_features=test_data[:,0]
test_labels=test_data[:,1]

train_features_updated=pr.generate_features(train_features,highest_degree)#Has all the powers upto 9
test_features_updated=pr.generate_features(test_features,highest_degree)

train_errors_ridge=np.zeros((2,lambda_size))#1st,2nd,3rd,4th columns shows train/test error for varying lambdas  for ridge /Lasso
test_errors_ridge=np.zeros((2,lambda_size))#1st and 2nd rows shows train/test errors for min error model and max errror models for ridge/Lasso
train_errors_lasso=np.zeros((2,lambda_size))#size is (2,4)
test_errors_lasso=np.zeros((2,lambda_size))

weights_for_ridge=np.zeros((2*lambda_size,highest_degree+1))#columns represent the coefficient of nth degree term
weights_for_lasso=np.zeros((2*lambda_size,highest_degree+1))#First two rows represent min error model and max error model respectively for lamda=0.25 and so on
if __name__=="__main__":
    print("_"*120)
    print("Learning rate        :\t",alpha)
    print("Termination limit    :\t",threshold)
    print("_"*120)   
    print(f"The polynomial which gives HIGHEST error is of degree:{M+1}")
    print(f"The polynomial which gives   LEAST error is of degree:{m+1}")
    print("The above statements are inferred from problem 1 in the assignment:Run main_file.py to view it")


##############################RiDGE#############################################################
if __name__=="__main__":
    print("_"*120)
    print(" "*45,"RIDGE REGRESSION"," "*40)
    print("_"*120)
    print("\tlambda\t\t\tPolynomial degree\t\tTrain error\t\tTest error\t\t\t\t\t")

for val in range(0,lambda_size):
    ind=0
    for deg in [m+1,M+1]:
        weights=np.zeros(deg+1)
        train_features_updated_now=train_features_updated[:,0:deg+1]
        test_features_updated_now=test_features_updated[:,0:deg+1]
        term=100
        while(term>threshold):#gradient descent step
        
            weights_updated=p3f.update_ridge(weights,train_features_updated_now,train_labels,alpha,lambdas[val])
            term=np.sum(np.square(weights-weights_updated))
            weights=weights_updated
            
        y_train_predicted=np.dot(train_features_updated_now,weights_updated)
        y_test_predicted=np.dot(test_features_updated_now,weights_updated)
        trainset_error_LS_ridge=p3f.LS_error(y_train_predicted,train_labels)
        testset_error_LS_ridge=p3f.LS_error(y_test_predicted,test_labels)
        for x in range(0,deg+1):
            weights_for_ridge[2*val+ind,x]=weights_updated[x]
        if __name__=="__main__":
            print("\t{0:}\t\t\t     \t{1:}\t\t\t{2:0.4f}\t\t\t{3:0.4f}\t\t\t\t\t".format(lambdas[val],deg,trainset_error_LS_ridge,testset_error_LS_ridge))
        train_errors_ridge[ind,val]=trainset_error_LS_ridge
        test_errors_ridge[ind,val]=testset_error_LS_ridge
        ind=ind+1
if __name__=="__main__":
    p3f.print_weights_p3(lambdas,weights_for_ridge,[m+1,M+1])
#################################LASSO############################################################333333333333333
if __name__=="__main__": 
    print("_"*120)
    print(" "*45,"LASSO REGRESSION"," "*40)
    print("_"*120)  
    print("\tlambda\t\t\tPolynomial degree\t\tTrain error\t\tTest error\t\t\t\t\t")     

for val in range(0,lambda_size):
    ind=0
    for deg in [m+1,M+1]:
        weights=np.zeros(deg+1)
        train_features_updated_now=train_features_updated[:,0:deg+1]
        test_features_updated_now=test_features_updated[:,0:deg+1]
        term=100
        while(term>threshold):#Gradient descent step
        
            weights_updated=p3f.update_lasso(weights,train_features_updated_now,train_labels,alpha,lambdas[val])
            term=np.sum(np.square(weights-weights_updated))
            weights=weights_updated
        
        for x in range(0,deg+1):
            weights_for_lasso[2*val+ind,x]=weights[x]
        
        #print(weights)
        y_train_predicted=np.dot(train_features_updated_now,weights_updated)
        y_test_predicted=np.dot(test_features_updated_now,weights_updated)
        trainset_error_LS_lasso=p3f.LS_error(y_train_predicted,train_labels)
        testset_error_LS_lasso=p3f.LS_error(y_test_predicted,test_labels)
        if __name__=="__main__":
            print("\t{0:}\t\t\t     \t{1:}\t\t\t{2:0.4f}\t\t\t{3:0.4f}\t\t\t\t\t".format(lambdas[val],deg,trainset_error_LS_lasso,testset_error_LS_lasso))
        
        train_errors_lasso[ind,val]=trainset_error_LS_lasso
        test_errors_lasso[ind,val]=testset_error_LS_lasso
        ind=ind+1
if __name__=="__main__":
    p3f.print_weights_p3(lambdas,weights_for_lasso,[m+1,M+1])
    p3f.plot_error_p3(train_errors_ridge,test_errors_ridge,train_errors_lasso,test_errors_lasso,lambdas,[m+1,M+1])
