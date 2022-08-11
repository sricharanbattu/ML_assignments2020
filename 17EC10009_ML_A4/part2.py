# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:06:18 2020

@author: lenovo
"""

import numpy as np
import pandas as pd
import sklearn

def accuracy(preds,actuals):
    """
    The arguments are one hot encoded vectors. 
    The proportion of correct predictions is given as output
    """
    x=preds*actuals
    n=preds.shape[0]
    return np.sum(x)/n

print("PART 2:")
trainset=np.genfromtxt('./data/train.csv',delimiter=',')
trainfeature=trainset[1:,:-4]
trainout=trainset[1:,7]
trainouts=trainset[1:,-3:]

testset=np.genfromtxt('./data/test.csv',delimiter=',')
testfeature=testset[1:,:-4]
testout=testset[1:,7]
testouts=testset[1:,-3:]

nn1=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32,),activation='logistic',learning_rate_init=0.01,batch_size=32)
nn2=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(64,32,),activation='relu',learning_rate_init=0.01,batch_size=32)

nn1.fit(trainfeature,trainouts)
trainpreds1=nn1.predict(trainfeature)
testpreds1=nn1.predict(testfeature)
print("_"*120)
print("                                     PART2 :SPECIFICATION 1A             ")
print("_"*120)
print(" final train accuracy          :                       {0:0.4f}".format(accuracy(trainpreds1,trainouts)))
print("final test accuracy           :                       {0:0.4f}".format(accuracy(testpreds1,testouts)))

nn2.fit(trainfeature,trainouts)
trainpreds2=nn2.predict(trainfeature)
testpreds2=nn2.predict(testfeature)
print("_"*120)
print("                                     PART2  :SPECIFICATION 1B             ")
print("_"*120)
print("final train accuracy          :                       {0:0.4f}".format(accuracy(trainpreds2,trainouts)))
print("final test accuracy           :                       {0:0.4f}".format(accuracy(testpreds2,testouts)))
print("_"*120)
