# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:18:33 2017

@author: v-yuewng
"""

from mnistcnn import mnistnet_cnn
import numpy as np
import matplotlib.pyplot as plt
model   = mnistnet_cnn(minibatchsize=64, learningrate = 0.01)


model.buildnet()

model.loaddata()

model.init_net()

model.data_mode(1)
model.train_mode(3)

epoch=0


acc = []
loss =[]


weight = []

grad_norm = []

dis =[]
model.lr = 0.01

for jj in range(50):
    model.eval_weight()
    weight.append(model.v_weight)
    model.lr *= 0.8
    for ii in range(10000): 

        model.global_step = 0
        model.next_batch()
    
    
        model.train_net( )
        
        if epoch < model.epoch:
            epoch = model.epoch
            model.save_model('cnn_exp1')
        
    
        if ii % 1000 == 0 :
            model.fill_train_data()
            model.calloss()
            loss.append(model.v_loss)
            
            model.calacc()
            acc.append(model.v_acc)
            print('epoch',model.epoch,'loss', model.v_loss,'acc',model.v_acc,'lr',model.lr)
            
        
    model.fill_train_data()
    model.eval_grad()
    print(model.v_grad_max)
    print(model.v_grad_min)
    print(model.v_grad_norm)
    grad_norm.append(model.v_grad_norm)
    
    model.eval_weight()
    weight.append(model.v_weight)
    
    dis_1 = np.linalg.norm(weight[-1]-weight[-2])
    dis.append(dis_1)   
    print(dis_1 )
