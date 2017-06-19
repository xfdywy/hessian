

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
            parameters =  tf.Variable(tf.concat(0, [tf.truncated_normal([5*5*3*64 ]), tf.zeros([64 ]),
                            tf.truncated_normal([5*5*64*64   ]), tf.zeros([64 ]),
                            tf.truncated_normal([1024*64   ]), tf.zeros([64 ]),
#                            tf.truncated_normal([384*192   ]), tf.zeros([192  ]),
                            tf.truncated_normal([64*10   ]), tf.zeros([10 ]),
                            ])   )      
            
            begin = 0
            para_con1 = tf.reshape(tf.slice(parameters,  [begin ],[5*5*3*64 ]), [5,5,3,64] )
            begin += 5*5*3*64
            para_con1_bias = tf.reshape(tf.slice(parameters,  [begin ],[64 ]), [64 ] )
            begin += 64
            
            para_con2 = tf.reshape(tf.slice(parameters,  [begin, ],[5*5*64*64 ]), [5,5,64,64] )
            begin += 5*5*64*64
            para_con2_bias = tf.reshape(tf.slice(parameters,  [begin, ], [64 ] ), [64 ])
            begin += 64
            
            para_fc3 = tf.reshape(tf.slice(parameters,  [begin ], [1024*64 ] ), [1024,64])
            begin += 1024*64
            para_fc3_bias = tf.reshape(tf.slice(parameters,  [begin ], [64 ] ), [64 ])
            begin += 64
            
#            para_fc4 = tf.reshape(tf.slice(parameters,  [begin ], [384*192 ] ), [384,192])
#            begin += 384*192
#            para_fc4_bias = tf.reshape(tf.slice(parameters,  [begin ], [192 ] ), [192 ])
#            begin += 192
            
            para_fc5 = tf.reshape(tf.slice(parameters,  [begin ], [64*10 ] ), [64,10])
            begin += 64*10
            para_fc5_bias = tf.reshape(tf.slice(parameters,  [begin ], [10 ] ), [10 ])
            begin += 10
            
            
            net = tf.nn.conv2d(self.images,para_con1,[1,1,1,1],'SAME',name='conv1') 
            net = tf.nn.bias_add(net,para_con1_bias)
            
            
#            net = slim.conv2d(self.images, 64, [5, 5], scope='conv1')

#            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')

#            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
            
            
            
            net = tf.nn.conv2d(net,para_con2,[1,1,1,1],'SAME',name='conv2') 
            net = tf.nn.bias_add(net,para_con2_bias)

#            net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
#            net = slim.max_pool2d(net, [4, 4], 4, scope='pool2')

            net = slim.flatten(net)
            net = tf.slice(net,[0,0],[-1,1024])

            net = tf.nn.relu(tf.matmul(net,para_fc3) + para_fc3_bias)

            net = tf.nn.dropout(x = net, keep_prob =  self.dropout_keep_prob , name='dropout3') 
# 
#            net = tf.nn.relu(tf.matmul(net,para_fc4) + para_fc4_bias)
#
#            net = tf.nn.dropout(x = net, keep_prob =  self.dropout_keep_prob , name='dropout4') 
            
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
        
 
