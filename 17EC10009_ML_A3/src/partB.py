# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:35:17 2020

@author: lenovo
"""

"""
This file is my solution to the Agglomerative clustering algorithm
"""
import numpy as np
import pandas as pd


def AgglomerativeClustering(data,n_clusters):
    """
    This function takes as arguments:
        1.data: A numpy array of data
        2.n_clusters: number of clusters
    This function initialises a cluster group matrix ,each row having the indices 
    a cluster.This gets updated at every level of Hierarchy. The last element of the 
    row keeps track of number of elements in the cluster.It finally outputs the
    sorted cluster list as required
    """
    n_rows=data.shape[0]
    cluster_group=np.zeros((n_rows,n_rows+1),dtype=int)                         #The array whose rows contain the indices of elements of a cluster,and whose last element shows the no of element 
    for i in range(0,n_rows):
        cluster_group[i,0]=i
        cluster_group[i,-1]=1
    cluster_group_obtained=get_clusters(data,cluster_group,n_clusters)          #get_clusters is a function returning the clusters
    cluster_list=[]
    for i in range(0,n_clusters):
        n=cluster_group_obtained[i,-1]
        cluster_list.append(list(cluster_group_obtained[i,:n]))
    return modified(cluster_list)                                               #returns the list of clusters sorted by the first element 
    
def get_clusters(data,cluster_group,n_clusters):
    """
    This function takes as arguments:
        1. data: A numpy array of dat
        2. cluster_group: a 2d numpy array whose rows has the index of the element 
        and last element as no of elements of that row.Cluster group is a numpy array which keeps track of what elements
        are present in a cluster.The argument is the cluster group at level0,which means each element is an individual
        cluster
        3. n_clusters: no of clusters required
    This function updates the cluster matrix in the hierarchy based on single linkage strategy and cosine similarity
    and returns the last 8 clusters ina matrix form
    """
    merged_up_clusters=[]                                                       #keeps track of which indices have been merged into another cluster
    n_rows_original=cluster_group.shape[0]                                      #keeps track of original number of individual clusters
    n_rows=cluster_group.shape[0]                                               #keeps track of how many clusters are present after a level in the hierarchy
    similarity_mat=similarity_matrix(data)                                      #keeps track of similarity matrix at each level in the hierarchy
    while(n_rows>8):                                                            #This loop represents the tasks done at each level in the hierarchy until n_clusters number of clusters form
        max_sim=np.amax(similarity_mat)                                         #stores the maximum value of similarity found from similarity matrix
        max_sim_index=np.where(similarity_mat==max_sim)[0]                      #stores which clusters have max similarity bw them
        if(cluster_group[max_sim_index[0],-1]<cluster_group[max_sim_index[1],-1]): #This if-else loop determines which cluster needs to be merged.The cluster with less no
            ToBeMergedCluster=max_sim_index[0]                                  #       is merged into the cluster with more no of elements.
            MergingCluster=max_sim_index[1]                                     #ToBeMergedCluster denotes the cluster which is destroyed
        else:                                                                   #MergingCluster denotes the cluster which will be preserved after merging
            MergingCluster = int(max_sim_index[0])
            ToBeMergedCluster=int(max_sim_index[1])
            
        n_g1=cluster_group[MergingCluster,-1]                                   #denotes no of elements in MergingCluster
        n_g2=cluster_group[ToBeMergedCluster,-1]                                #denotes no of elements in the cluster that will get destroyed
        
        for i in range(n_g1,n_g1+n_g2):
            cluster_group[MergingCluster,i]=cluster_group[ToBeMergedCluster,i-n_g1] #the elements of one cluster are added to the cluster that gets preserved
        cluster_group[MergingCluster,-1]=n_g1+n_g2                              #the no of elements of the preserved cluster are updated
        cluster_group[ToBeMergedCluster,-1]=0
        for i in range(0,n_rows_original):                                      #This loop updates the similarity matrix
            similarity_mat[MergingCluster,i]=max(similarity_mat[MergingCluster,i],similarity_mat[ToBeMergedCluster,i])
            similarity_mat[i,MergingCluster]=similarity_mat[MergingCluster,i]
            similarity_mat[MergingCluster,MergingCluster]=0
        for i in range(0,n_rows_original):                                       #This loop sets the destroyed cluster similarities with other clusters as 0.Hence this cluster is deactivated from next levels
            similarity_mat[ToBeMergedCluster,i]=0
            similarity_mat[i,ToBeMergedCluster]=0
        n_rows=n_rows-1                                                         #updates the no of clusters
        merged_up_clusters.append(ToBeMergedCluster)                            #takes note of the destroyed clusters for deletion                            
    cluster_group=np.delete(cluster_group,merged_up_clusters,axis=0)            #deletes the destroyed clusters
    return cluster_group
    

    
    


def similarity_matrix(data):
    """
    This function takes as arguments:
        1.data : A numpy array of data
    This function constructs a two dimensional matrix whose i,j element determines
    the cosine similarity of data[i,:] and data[j,:]
    """
    n=data.shape[0]
    sim=np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            if(i!=j):
                sim[i,j]=cosine_sim(data[i,:],data[j,:])
                
    return sim
            


def inv_cosine_similarity(data1,data2):
    """
    This function takes as arguments:
        1.data1: a data vector
        2.data2: another datavector
    This function returns inverse cosine similarity,a distance metric
    """
    return np.exp(-1*np.dot(data1,data2))

def cosine_sim(data1,data2):
    """
    This function takes as arguments:
        1.data1: a data vector
        2.data2: another datavector
    This function returns  cosine similarity,a similarity  metric
    """
    return np.dot(data1,data2)

def modified(clusterlist):
    """
    This function takes as  arguments:
        1. The list of clusters each element containing a list of indices:cluster_matrix
    This function arranges the list of clusters, based on the first element 
    of ordered clusters.
    """
    arr=[]
    sh=len(clusterlist)
    for i in range(0,sh):
        arr.append(sorted(clusterlist[i]))
        
    arr1=[]
    for i in range(0,sh):
        x=arr[i]
        lis=[i,x[0]]
        arr1.append(lis)
    arr1.sort(key=lambda lis:lis[1])
   
    arr2=[]
    for i in range(0,sh):
        arr2.append(arr[arr1[i][0]])
    return arr2

