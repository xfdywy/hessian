# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:31:23 2017

@author: v-yuewng
"""
import theano
import theano.tensor as T
from theano import function
import numpy 
import matplotlib.pyplot as plt
x,z = T.dvectors('x','z')
y = x ** 2 + z**2 

cost= T.max(y)
gy =T.grad(cost, [x,z])  [0]

H,updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x),sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])  
f =function([x], H, updates=updates)  

#
#f([2,4])

"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy as np
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano import pp




import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



class lenet:
    def __init__(self,learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',input_shape=[3,32,32],num_classes=10,
                    nkerns=[20, 50], batch_size=50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.nkerns = nkerns
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def buildmodel(self):
        """ Demonstrates lenet on MNIST dataset
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    
        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)
    
        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """

        rng = numpy.random.RandomState(23455)
    
 
        model = Sequential()
        
        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.input_shape,data_format=  "channels_first"))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3),padding='same',data_format=  "channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same',data_format=  "channels_first"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same',data_format=  "channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        
        self.predict = model.output
        self.x = model.input
        self.y =  K.placeholder([None],dtype='int32')
#        loss = K.categorical_crossentropy(self.predict,self.y).mean()
        y =  K.one_hot(self.y,10)
        loss = K.mean(( y - self.predict)**2)
        
        paras = model.weights
        
        self.dl_dw = K.gradients(loss,paras)
        self.x = paras 
 
        
        self.all_paras_dim = [x.shape.eval() for x in self.x]
        self.all_paras_num = [np.prod(x) for x in self.all_paras_dim]
            
            
        
        
        H,updates = theano.scan(lambda i, y,x : T.grad(y[i][0], x), sequences=T.arange(3), non_sequences=[y, x])
        
        H_fun = K.function([self.x,self.y,K.learning_phase()],H)
        
        
        pred = K.function([self.x,self.y,K.learning_phase()],self.predict)
        f = K.function([self.x,self.y,K.learning_phase()],dl_dw)
        
    def cal_allvar_list(self):
        self.all_d_list = [None]*
        for nii,ii in enumerate(self.all_paras_dim):
            print(ii)
            if len(ii)==4:
                for d_1 in range(ii[0]):
                    for d_2 in range(ii[1]):
                        for d_3 in range(ii[2]):
                            for d_4 in range(ii[3]):
                                self.all_d_list.append(self.dl_dw[nii][d_1][d_2][d_3][d_4])
            elif len(ii)==2:
                for d_1 in range(ii[0]):
                    for d_2 in range(ii[1]):
 
                                self.all_d_list.append(self.dl_dw[nii][d_1][d_2] )
            elif len(ii)==1:
                for d_1 in range(ii[0]):
                    self.all_d_list.append(self.dl_dw[nii][d_1]  )
        
        