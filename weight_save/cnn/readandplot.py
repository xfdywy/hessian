import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

all_file = os.listdir('./')
all_file = [x for x in all_file if 'pkl' in x]
res = {}
for ii in all_file:
    alg = ii.split('_')[0]
    with open(ii,'rb') as f:
        if alg in res.keys() : 
            res[alg].append( pickle.load(f))
        else :
            res[alg] = [pickle.load(f)]
    
        
a = [x for x in res.values()]

all_w = []

for ii in a:
    for jj in ii:
        all_w.append(jj)

dis_mat = np.zeros([6,6])       
corr_mat = np.zeros([6,6])

for nii,ii in enumerate(all_w):
    for njj,jj in enumerate(all_w):
        dis_mat[nii,njj] = np.linalg.norm(ii-jj)
#        corr_mat[nii,njj] = np.corrcoef(ii,jj)