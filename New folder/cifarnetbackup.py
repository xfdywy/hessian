# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:55:34 2017

@author: v-yuewng
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:45:50 2017

@author: v-yuewng
"""

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the CIFAR-10 model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)
import numpy as np
class cifarnet:
    def __init__(self,num_classes=10,minibatchsize=1,imagesize=32,dropout_keep_prob=1 ,scope='cifarnet' ,learningrate = 0.001):
       self.num_classes=num_classes  
       self.minibatchsize=minibatchsize
       self.imagesize = imagesize
#       self.dropout_keep_prob=dropout_keep_prob
       self.scope=scope 
       self.prediction_fn=slim.softmax
       self.is_training = True
#       self.learningrate = learningrate
  

    def buildnet(self):
#        self.end_points = {}
        
        self.learningrate = tf.placeholder('float32',[ ])
        self.images = tf.placeholder('float32',[None,self.imagesize,self.imagesize,3])
        self.label = tf.placeholder('int32',[None,])
        self.dropout_keep_prob = tf.placeholder('float32',[])
        with tf.variable_scope(self.scope, 'CifarNet', [self.images, self.num_classes]):
            parameters =  tf.Variable(tf.concat(0, [tf.truncated_normal([5*5*3*64, 1]), tf.zeros([64, 1]),
                            tf.truncated_normal([5*5*64*64  ,1]), tf.zeros([64, 1]),
                            tf.truncated_normal([4096*384  ,1]), tf.zeros([384, 1]),
                            tf.truncated_normal([384*192  ,1]), tf.zeros([192, 1]),
                            tf.truncated_normal([192*10  ,1]), tf.zeros([10, 1]),
                            ])   )      
            
            begin = 0
            para_con1 = tf.reshape(tf.slice(parameters,  [begin,0],[5*5*3*64,1]), [5,5,3,64] )
            begin += 5*5*3*64
            para_con1_bias = tf.reshape(tf.slice(parameters,  [begin,0],[64,1]), [64 ] )
            begin += 64
            
            para_con2 = tf.reshape(tf.slice(parameters,  [begin,0],[5*5*64*64,1]), [5,5,64,64] )
            begin += 5*5*64*64
            para_con2_bias = tf.reshape(tf.slice(parameters,  [begin,0], [64,1] ), [64 ])
            begin += 64
            
            para_fc3 = tf.reshape(tf.slice(parameters,  [begin,0], [4096*384,1] ), [4096,384])
            begin += 4096*384
            para_fc3_bias = tf.reshape(tf.slice(parameters,  [begin,0], [384,1] ), [384 ])
            begin += 384
            
            para_fc4 = tf.reshape(tf.slice(parameters,  [begin,0], [384*192,1] ), [384,192])
            begin += 384*192
            para_fc4_bias = tf.reshape(tf.slice(parameters,  [begin,0], [192,1] ), [192 ])
            begin += 192
            
            para_fc5 = tf.reshape(tf.slice(parameters,  [begin,0], [192*10,1] ), [192,10])
            begin += 192*10
            para_fc5_bias = tf.reshape(tf.slice(parameters,  [begin,0], [10,1] ), [10 ])
            begin += 10
            
            
            net = tf.nn.conv2d(self.images,para_con1,[1,1,1,1],'SAME',name='conv1') 
            net = tf.nn.bias_add(net,para_con1_bias)
            
            
#            net = slim.conv2d(self.images, 64, [5, 5], scope='conv1')

            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')

            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
            
            
            
            net = tf.nn.conv2d(net,para_con2,[1,1,1,1],'SAME',name='conv2') 
            net = tf.nn.bias_add(net,para_con2_bias)

            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

            net = slim.flatten(net)

            net = tf.nn.relu(tf.matmul(net,para_fc3) + para_fc3_bias)

            net = tf.nn.dropout(x = net, keep_prob =  self.dropout_keep_prob , name='dropout3') 
 
            net = tf.nn.relu(tf.matmul(net,para_fc4) + para_fc4_bias)

            net = tf.nn.dropout(x = net, keep_prob =  self.dropout_keep_prob , name='dropout4') 
            
            self.logits = tf.matmul(net,para_fc5) + para_fc5_bias
            
#            self.logits = slim.fully_connected(net, self.num_classes,
#                                      biases_initializer=tf.zeros_initializer(),
#                                      weights_initializer=trunc_normal(1/192.0),
#                                      weights_regularizer=None,
#                                      activation_fn=None,
#                                      scope='logits')
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,labels = self.label)
            self.meanloss = tf.reduce_mean(self.loss)

#            self.end_points['Predictions'] = self.prediction_fn(self.logits, scope='Predictions')
            
            self.allvars = tf.trainable_variables()
            self.init_allvars = tf.variables_initializer(self.allvars)
            self.train = tf.train.GradientDescentOptimizer(self.learningrate).minimize(self.meanloss)
            
    def calacc(self,sess,datax,datay,dropout_prob):
        predict = sess.run(self.logits,feed_dict = {self.images : datax , self.label : datay ,self.dropout_keep_prob : dropout_prob})
        predict = np.argmax(predict,1)
        return(np.sum(predict == datay)*1.0 / len(datay))
        
        
  
  
#  
#  
#cifarnet.default_image_size = 32
#
#
#def cifarnet_arg_scope(weight_decay=0.004):
#  """Defines the default cifarnet argument scope.
#
#  Args:
#    weight_decay: The weight decay to use for regularizing the model.
#
#  Returns:
#    An `arg_scope` to use for the inception v3 model.
#  """
#  with slim.arg_scope(
#      [slim.conv2d],
#      weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
#      activation_fn=tf.nn.relu):
#    with slim.arg_scope(
#        [slim.fully_connected],
#        biases_initializer=tf.constant_initializer(0.1),
#        weights_initializer=trunc_normal(0.04),
#        weights_regularizer=slim.l2_regularizer(weight_decay),
#        activation_fn=tf.nn.relu) as sc:
#      return sc
