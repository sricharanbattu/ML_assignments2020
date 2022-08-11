# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:38:04 2020

@author: lenovo
"""
import numpy as np
import pandas as pd
import partE as pE

"""
This file is my solution to KMeans clustering algorithm.
A centroid array(of dimensions n_clustersxn_columns) is maintained,each row
indicating a centroid.
An array ofdimensions n_rowsxn_clusters is maintained,each row indicating 
distances of the particular instance from all centroids.
An array of dim (1xn_rows),giving the cluster identity for each instance,is also 
maintained throughout the algorithm

"""

def KMeansClustering(data,n_clusters):
    """
    This function takes  as arguments:
        1.A numpy array of data
        2.no of clusters n_clusters
    This function initialises the centroids in a random fashion and then 
    finds the cluster identity of each instance.
    These two steps of finding centroids and clusters are done iteratively.
    It, then returns the clusters each arranged according to the ascending order of the indices
    """
    
    centroids=Initialise_Centroids(data,n_clusters)                             #centroids are randomly initialised
    old_cluster_matrix=get_clusters(data,centroids)                             #the first set of clusters are found based on initial centroids
    
    
    while(True):                                                                #E step and M step are performed iteratively until the clusters don't change
       
       centroids=get_centroids(data,old_cluster_matrix,n_clusters)
       cluster_matrix=get_clusters(data,centroids)
       diff=np.sum(cluster_matrix-old_cluster_matrix)
       if(diff==0):                                                             #if the clusters don't change,break from the loop,otherwise store the present clusters for future comparisons
           break
       else:
           old_cluster_matrix=cluster_matrix
        
    return modified(cluster_matrix,n_clusters)                                  #return the list of data indices,each element referring to a cluster and the indices are in ascending order
   

def Initialise_Centroids(data,n_clusters):    
    """
    Initialise the centroids based on the no of clusters randomly.
    """                                  
    n_cols=data.shape[1]
    centroids=np.zeros((n_clusters,n_cols))
    
    low=np.amin(data)
    high=np.amax(data)
    
    
    for i in range(0,n_clusters):
        centroids[i,:]=np.random.uniform(low=low,high=high,size=(n_cols,))      #the initialisation is done such that each element is taken randomly from a uniform distribution
                                                                                #the lies between minimum and max elements of the data
    return centroids



def get_clusters(data,centroids):   
    """
    This functions take as arguments:
        1. A numpy array of data
        2. The centroids that give clusters
    When the centroids are given, each instance looks at the nearest centroid
    and assigns itself to the cluster represented by nearest centroid
    """                                            
    n_clusters=centroids.shape[0]
    n_rows=data.shape[0]
    similarity_matrix=np.zeros((n_rows,n_clusters))
    cluster_matrix_=np.zeros(n_rows,dtype=int)
    for i in range(0,n_rows):
        for j in range(0,n_clusters):
            similarity_matrix[i,j]=cosine_similarity(data[i,:],centroids[j,:])
    
    cluster_matrix_=np.argmax(similarity_matrix,axis=1)                           #The nearest centroid is found based on similarity bw instance and centroid
    return cluster_matrix_


def get_centroids(data,cluster_matrix,n_clusters):
    """
    This function takes as arguments:
        1. a numpy array of data
        2. The matrix showing the cluster of each instance:cluster_matrix
        3. no of clusters:n_clusters
    
    the average of data points belonging to a particualr cluster is taken as centroid of 
    that cluster
    """
    n_cols=data.shape[1]
    centroids_=np.zeros((n_clusters,n_cols))
    for i in range(0,n_clusters):
        truth=(cluster_matrix==i)
        centroids_[i,:]= np.dot(truth,data)/np.sum(truth)
    return centroids_
        

    
def cosine_similarity(vector1,vector2):
    """
    This function takes as arguments:
        1. a  data vector vector1
        2.second data vector vector2
    The similarity bw two vectors is defined as the dot product of normalised vectors
    """
    return np.dot(vector1,vector2)

def inverse_cosine_similarity(vector1,vector2):
    return np.exp(-1*np.dot(vector1,vector2))


def modified(cluster_matrix,n_clusters):
    """
    This function takes as arguments:
        1. The list of clusters each element containing a list of indices:cluster_matrix
        2. number of clusters : n_clusters
    This function arranges the list of clusters, based on the first element 
    of ordered clusters.
    """
    arr=[]                                                                      #The first array arr[] sorts each element of the list
    for i in range(0,n_clusters):
        indices=list(np.where(cluster_matrix==i)[0])
        lis=sorted(indices)
        arr.append(lis)
        
    arr1=[]                                                                     #The second array arr1[] takes first element and the index of the element
    for i in range(0,n_clusters):                                               #And then sorts arr1[] based on first element
        x=arr[i]
        lis=[i,x[0]]
        arr1.append(lis)
    arr1.sort(key=lambda lis:lis[1])
   
    arr2=[]                                                                     #The third array arr2[] gets the sorted list 
    for i in range(0,n_clusters):
        arr2.append(arr[arr1[i][0]])
    return arr2
    
    
"""data=np.genfromtxt('unlabelled_m.csv',delimiter=',')
data=data[1:,1:]
n_clusters=8
cluster_list=KMeansClustering(data,n_clusters)

datal=pd.read_csv('labelled_m.csv')
dataframe=datal['Religious texts']
print(pE.NMI(cluster_list,dataframe))
f=open('kmeans.txt','w')
for i in range(0,n_clusters):
    x=list(cluster_list[i])
    l=len(x[0])
    for j in range(0,l):
        f.write(str(int(x[0][j])))
        if(j<l-1):
            f.write(',')
    f.write('\n')
    
f.close()
"""