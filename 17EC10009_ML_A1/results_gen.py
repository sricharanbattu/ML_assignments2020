# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:47:20 2020

@author: sricharanbattu
Name: BATTU SRICHARAN
Roll No: 17EC10009
email id: sricharanbattu@gmail.com
"""

import part3 as p3

import poly_regression as pr
import part3_functions as p3f

import numpy as np
import matplotlib.pyplot as plt
import sys

alpha=p3.alpha
threshold=p3.threshold
weight=p3.weight_polyreg
error_test=p3.testerror_polyreg
error_train=p3.trainerror_polyreg
degree=p3.highest_degree
lambdas=p3.lambdas
train_errors_ridge=p3.train_errors_ridge
test_errors_ridge=p3.test_errors_ridge
train_errors_lasso=p3.train_errors_lasso
test_errors_lasso=p3.test_errors_lasso

m=p3.m
M=p3.M
weights_for_ridge=p3.weights_for_ridge
weights_for_lasso=p3.weights_for_lasso

f=open('result.txt',"w") 

f.write("_"*120+"\n")
f.write("Learning rate        :\t"+str(alpha)+"\n")
f.write("Termination limit    :\t"+str(threshold)+"\n")
f.write("_"*120+"\n")  
f.write("_"*60+"PART 1 RESUTS"+"_"*50+"\n")

f.write("The weights obtained for the different degree polynomial regression model are as follows:\n\n")
pr.print_weights(weight,list(range(0,9)),f)   
f.write("The errors obtained for the different degree polynomial regression model are as follows:\n\n")  
pr.print_errors(error_test,error_train,list(range(1,degree+1)),f) 
f.write("_"*60+"PART 2 RESUTS"+"_"*50+"\n")
f.write(f"1.The polynomial which gives HIGHEST error is of degree:{M+1}\n")
f.write(f"2.The polynomial which gives   LEAST error is of degree:{m+1}\n")
f.write("The above statements are inferred from part 1 in the assignment.\n\n") 
f.write(f"RESULT:Based on the errors in Part 1, It could be inferred that {m+1}th degree polynomial regression model is more suitable for this problem as it has both train and test errors minimum.\n") 
f.write("_"*60+"PART 3 RESUTS"+"_"*50+"\n")
f.write("_"*120+"\n")
f.write(" "*45+"RIDGE REGRESSION"+" "*40+"\n")
f.write("_"*120+"\n")
p3f.print_weights_p3(lambdas,weights_for_ridge,[m+1,M+1],f)
p3f.print_errors_p3(lambdas,train_errors_ridge,test_errors_ridge,[m+1,M+1],f)
f.write("_"*120+"\n")
f.write(" "*45+"LASSO REGRESSION"+" "*40+"\n")
f.write("_"*120+"\n")
p3f.print_weights_p3(lambdas,weights_for_lasso,[m+1,M+1],f)
p3f.print_errors_p3(lambdas,train_errors_lasso,test_errors_lasso,[m+1,M+1],f)
f.write("Based on  test errors observed,It is preferable to use LASSO for this problem.\n")
f.write("It is noteworthy that both LASSO and RIDGE regressions show more training error than Polynomial Regression and errors are actually increasing with lambda.It could be seen that the coefficient magnitudes are decreasing with lambda as a whole.\n")
f.write("_"*120+'\n')
f.close()     
    