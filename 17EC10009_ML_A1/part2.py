# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:05:31 2020

@author: sricharan battu
NAME      : BATTU SRI CHARAN
ROLL NO   : 17EC10009
gmail id  :sricharanbattu@gmail.com
"""
import main_file as mainf

import part2_functions as p2f

import numpy as np
import matplotlib.pyplot as plt


w=mainf.weight
etrain=mainf.error_train
etest=mainf.error_test
deg=mainf.degree
all_features=mainf.B_train_big
labels=mainf.labels_train

for i in range(0,deg):
    p2f.plot_line_curve(all_features[:,:i+2],w[i,:i+2],labels,i+1)

p2f.plot_error(etrain,etest,deg)
    
