# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:42:30 2020

@author: lenovo
"""
import pandas as pd
import numpy as np

def entropy(data,target_column):
    """
    Arguments::
        1)data: A dataframe 
        2)target_column:The attribute which needs to be predicted
    functionality::
        Calculates the entopy of the distribution of data[target_column]
        
    """
    x       =   list(data[target_column].unique())
    total   =   data.shape[0]
    entropy =   0
    for val in x:
        no_of_val   =   data.loc[data[target_column]==val].shape[0]
        p           =   no_of_val/total
        entropy     =   entropy-p*np.log2(p) 
    return entropy

def best_split_attribute(data,feature_columns,target_column):
    """
    Arguments::
        1)data: A dataframe
        2)feature_columns: The attributes of data based on which 
        the predictions are made
        3)target_column: The attribute of data which needs to be 
        predicted
    functionality::
        chooses which attribute decreases the maximum entropy at a given step
    """
    entropy_list=np.zeros(len(feature_columns))
    length=len(feature_columns)
    for i in range(0,length):
        col  =  feature_columns[i]
        x    =  list(data[col].unique())
        ent=0
        for val in x:
            dat=data[data[col]==val]
            ent=ent+entropy(dat,target_column)*(dat.shape[0]/data.shape[0])
        entropy_list[i]=ent
    
    entropy_list=list(entropy_list)
    if(len(entropy_list)==0):
        return None
    min_ind=entropy_list.index(min(entropy_list)) 
    
    return feature_columns[min_ind]   
    

class Node:
    """
    This represents the Node in the decision Tree.
    It has the member variables as::
        1)NodeType:whether it is a leaf or intermediary node/Root node in DecisionTree
    
        2)split_attribute:which attribute should be used to split at this stage
    
        3)reach_type: How has the instance to be predicted reach this particular leaf
        i.e why is the further expansion into the tree stopped.
        If it is an intermediary node it doesn't mean anything and is given a value of 4
        
        4)data_len:Number of training instances that reached this node
    
        5)major_class: which class dominates the data .Used to assign the class at the leaf
    """
    
    
    def __init__(self,data,feature_columns,target_column):
        self.value_counts=data[target_column].value_counts()
        if(data.shape[0]<=10 and data.shape[0]>0):                              #If the amount of  train data for further splitting at this node is very small,it is a leaf
            self.data_len=data.shape[0]
            self.NodeType=2
            self.major_class=int(data[target_column].value_counts().idxmax())
            self.split_attribute=None
            self.reach_type=0
        elif(data.shape[0]==0):                                                 #No Train data at this node.Consider it a leaf
            self.data_len=data.shape[0]
            self.NodeType=2
            self.split_attribute=None
            self.major_class=1
            self.reach_type=1
        elif(len(data[target_column].unique().tolist())==1):                    #All the data is pure.Consider it as a leaf
            self.data_len=data.shape[0]
            self.NodeType=2
            self.major_class=int(data[target_column].value_counts().idxmax())
            self.split_attribute=None
            self.reach_type=2
        elif(len(feature_columns)==0):                                          #No feature is left to be used to split.Make this node as a leaf
            self.data_len=data.shape[0]
            self.NodeType=2
            self.major_class=int(data[target_column].value_counts().idxmax())
            self.split_attribute=None
            self.reach_type=3
        else:                                                                   #All other nodes are intermediary nodes and has to be further splitted
            self.data_len=data.shape[0]
            self.split_attribute=best_split_attribute(data,feature_columns,target_column)
            self.NodeType=1
            self.major_class=int(data[target_column].value_counts().idxmax())
            self.reach_type=4
            
    def print_Node(self):
        print("No of data rows: ",self.data_len,"  Node Type: ",self.NodeType)
        print("Major Class:     ",self.major_class," split_attribute: ",self.split_attribute)
        print("Value counts: ",self.value_counts)
        print("Reach Type  :",self.reach_type)
            
        
        
class DecisionTree:
    """
    This class is a decision tree construction,essentially representing a subtree of the main tree
    It has its member variables as::
        1)RootNode: A node to start with
        2)data,feature_columns,target_column: training data to be considered for splitting
        3)level : At which level is the subtree at in the main tree
        4)Subtree0 : The child tree taking all the data of the data in the present tree,for which the split attribute 
        takes the value 0.
        5)Subtree1,Subtree2,Subtree3 are similar to Subtree0 but here the split attribute takes
        values 1,2,3 respectively.
    It also has a member funtion,which takes as input the data to be predicted and traverses the decision tree
    and stops at a leaf node.
    """
    def __init__(self,data,feature_columns,target_column,level=0):
        self.RootNode=Node(data,feature_columns,target_column)
        self.data=data
        self.feature_columns=feature_columns
        self.target_column=target_column
        self.level=level
        if(self.RootNode.NodeType==2):
            
            return
        else:
            spl_att=self.RootNode.split_attribute                               
            feature_col=feature_columns[:]
            feature_col.remove(spl_att)
            target_col=target_column
            data0=data[data[spl_att]==0]
            data1=data[data[spl_att]==1]
            data2=data[data[spl_att]==2]
            data3=data[data[spl_att]==3]
            if(data0.shape[0]>=0):
                self.SubTree0=DecisionTree(data0,feature_col,target_col,self.level+1)
            if(data1.shape[0]>=0):
                self.SubTree1=DecisionTree(data1,feature_col,target_col,self.level+1)
            if(data2.shape[0]>=0):
                self.SubTree2=DecisionTree(data2,feature_col,target_col,self.level+1)
            if(data3.shape[0]>=0):
                self.SubTree3=DecisionTree(data3,feature_col,target_col,self.level+1)
        
    
    def predict(self,data):                                                         
        if(self.RootNode.NodeType==2):                                              #If root node is reached during tree traversal, assign the majority class of this node as output
            
            x= self.RootNode.major_class
            return x
        else:                                                                       #If an intermediary node is reached during tree traversal,traverse further based on the split attribute
            att=self.RootNode.split_attribute
            
            if(int(data[att])==0):
                x=self.SubTree0.predict(data)
            elif(int(data[att])==1):
                x=self.SubTree1.predict(data)
            elif(int(data[att])==2):
                x=self.SubTree2.predict(data)
            elif(int(data[att])==3):
                x=self.SubTree3.predict(data)
            else:
                x=1                                                                 #Predict the bigger class as a buffer incase some erroneous condition occured
            return x

