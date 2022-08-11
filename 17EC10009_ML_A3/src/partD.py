# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:36:25 2020

@author: lenovo
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
"""
This file is my solution to partd of the question,
It performs dimensionality reduction using PCA
The reduced components are stored in a csv file named'unlabelled_m_reduced.csv'
 for further use
 The ith component is named PCi

"""

n=100
cols=[]
for i in range(0,n):
    cols.append("PC"+str(i+1))
    
    
unlabelled_m = np.genfromtxt('../data/unlabelled_m.csv',delimiter=',')
unlabelled_m=unlabelled_m[1:,1:]


pca = PCA(n_components=n)
principleComponents = pca.fit_transform(unlabelled_m)

unlabelled_m_reduced = pd.DataFrame(principleComponents,columns=cols)
unlabelled_m_reduced.to_csv('../data/unlabelled_m_reduced.csv')