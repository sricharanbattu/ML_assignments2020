# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
This provides solution to part A of the question

"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

labelled = pd.read_csv('../data/labelled.csv')     #the labelled file is read

n_rows=labelled.shape[0]
for i in range(0,n_rows):                                                       #The religious text of each row is renamed
    text=str(labelled.iloc[i,0])    
    ind=text.find('_')                                                          #By finding the index of underscore in the given text,the name is truncated upto that index
    text=text[:ind]
    labelled.iloc[i,0]=text                                                     #the text name is replaced
    
    
labelled.rename(columns={labelled.columns[0]:"Religious texts"},inplace=True)   #The unnamed column is named as 'Religious texts'
cluster_names=list(labelled['Religious texts'].unique())
labelled.drop(labelled.index[12],inplace=True)                                  #The trivial instance is deleted
labelled.to_csv('../data/labelled_m.csv')                                               #a new csv file with replaced texts is made


"""                     TF-IDF MATRIX                           """
cols = list(labelled.columns)
col_len=len(cols)
unlabelled_data=np.genfromtxt('../data/labelled.csv',delimiter=',')
unlabelled_data=unlabelled_data[1:,1:]
unlabelled_data=np.delete(unlabelled_data,12,axis=0)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True,norm='l2')
unlabelled_marray=tfidf_transformer.fit_transform(unlabelled_data).toarray()
unlabelled_m=pd.DataFrame(unlabelled_marray,columns=cols[1:])
unlabelled_m.drop(unlabelled_m.index[12])
unlabelled_m.to_csv('../data/unlabelled_m.csv')
 
