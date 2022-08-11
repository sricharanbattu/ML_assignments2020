# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:15:44 2020

@author: lenovo
"""

import pandas as pd

main_data=pd.read_csv('../data/winequality-red.csv',sep=';')
main_col=main_data.columns
quality=main_col[-1]

"""                         PREPROCESSING FOR LOGISTIC REGRESSION                   """
data1=main_data
data1.loc[data1[quality]<=6,quality]=0;
data1.loc[data1[quality]>6,quality]=1;
for col in main_col[:-1]:
    min=data1[col].min()
    max=data1[col].max()
    data1[col]=(data1[col]-min)/(max-min)
data1.to_csv('../data/datasetA.csv')

"""                         PREPROCESSING FOR DECISION TREES                   """
main_data=pd.read_csv('../data/winequality-red.csv',sep=';')
main_col=main_data.columns
quality=main_col[-1]
data2=main_data
data2.loc[data2[quality]<5,quality]=0;
data2.loc[(data2[quality]>=5) & (data2[quality]<=6),quality]=1;
data2.loc[data2[quality]>6,quality]=2;
for col in main_col[:-1]:
    mean1=data2[col].mean()
    std1=data2[col].std()
    data2[col]=(data2[col]-mean1)/std1
    
temp='temp'
for col in main_col[:-1]:    
    min1=data2[col].min()
    max1=data2[col].max()
    diff=(max1-min1)/4
    partition1=min1+diff
    partition2=partition1+diff
    partition3=partition2+diff
    data2[temp]=0
    data2.loc[(data2[col]>=min1) & (data2[col]<partition1),temp]=0
    data2.loc[(data2[col]>=partition1) & (data2[col]<partition2),temp]=1
    data2.loc[(data2[col]>=partition2) & (data2[col]<partition3),temp]=2
    data2.loc[(data2[col]>=partition3) & (data2[col]<=max1),temp]=3
    data2[col]=data2[temp]
    
del data2['temp']
data2.to_csv('../data/datasetB.csv')
    


