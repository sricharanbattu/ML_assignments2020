# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
"""
This proccesses the crude data file given for the assignment
"""

file_old=open('./data/dataset_raw.txt','r')
file_new=open('./data/datasetnew.txt','w')
for lines in file_old:
    a=lines.replace('\t\t','\t')
    file_new.write(a+'\n')
file_old.close()
file_new.close()
    

