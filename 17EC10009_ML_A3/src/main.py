# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:37:07 2020

@author: lenovo
"""

import partB as pB
import partC as pC
import partE as pE
import numpy as np
import pandas as pd
"""                            READING DATA AND ACTUAL CLUSTERS FOR EVALUATION             """

data=np.genfromtxt('../data/unlabelled_m.csv',delimiter=',')
data=data[1:,1:]

data_reduced=np.genfromtxt('../data/unlabelled_m_reduced.csv',delimiter=',')
data_reduced=data_reduced[1:,1:]

datal=pd.read_csv('../data/labelled_m.csv')
dataframe=datal['Religious texts']
n_clusters=8

"""                             FINDING CLUSTERS BY DIFFERENT ALGORITHMS                            """

Agglist=pB.AgglomerativeClustering(data,n_clusters)
kmeanslist=pC.KMeansClustering(data,n_clusters)
Agglistr=pB.AgglomerativeClustering(data_reduced,n_clusters)
kmeanslistr=pC.KMeansClustering(data_reduced,n_clusters)

"""                             PRINTING THE NMI METRICS                                    """

print('_'*120)
AggNMIur=pE.NMI(Agglist,dataframe)
print("NMI for Agglomerative Clustering unreduced   =       ",AggNMIur)

kmeansNMIur=pE.NMI(kmeanslist,dataframe)
print("NMI for K means Clustering unreduced         =        ",kmeansNMIur)

AggNMIr=pE.NMI(Agglistr,dataframe)
print("NMI for Agglomerative Clustering reduced     =        ",AggNMIr)

kmeansNMIr=pE.NMI(kmeanslistr,dataframe)
print("NMI for K means Clustering reduced           =        ",kmeansNMIr)
print('_'*120)

"""                             WRITING THE CLUSTERS TO VARIOUS FILES                       """
f=open('../clusters/agglomerative.txt','w')
for i in range(0,n_clusters):
    cluster=Agglist[i]
    l=len(cluster)
    for j in range(0,l):
        f.write(str(cluster[j]))
        if(j<l-1):
            f.write(',')
    f.write('\n')
f.close()
    
f=open('../clusters/kmeans.txt','w')
for i in range(0,n_clusters):
    cluster=kmeanslist[i]
    l=len(cluster)
    for j in range(0,l):
        f.write(str(cluster[j]))
        if(j<l-1):
            f.write(',')
    f.write('\n')
f.close()

f=open('../clusters/agglomerative_reduced.txt','w')
for i in range(0,n_clusters):
    cluster=Agglistr[i]
    l=len(cluster)
    for j in range(0,l):
        f.write(str(cluster[j]))
        if(j<l-1):
            f.write(',')
    f.write('\n')
f.close()
    
f=open('../clusters/kmeans_reduced.txt','w')
for i in range(0,n_clusters):
    cluster=kmeanslistr[i]
    l=len(cluster)
    for j in range(0,l):
        f.write(str(cluster[j]))
        if(j<l-1):
            f.write(',')
    f.write('\n')
f.close()