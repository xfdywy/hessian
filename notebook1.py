# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 21:39:06 2017

@author: v-yuewng
"""
from mnistdnn import cifarnet



a = cifarnet(minibatchsize=10, learningrate = 0.0001)

a.buildnet()

a.loaddata()

a.init_net()

a.data_mode(2)
a.train_mode(1)

a.next_batch()


a.train_net( )

a.fill_test_data()
a.calloss()
print(a.v_loss)


a.calacc()
print(a.v_acc)


self = a
feed_dict = {self.images : self.datax, self.label : self.datay ,
                     self.learningrate : self.lr , self.dropout_keep_prob : self.dp,
                     self.momentum : self.mt}
                     
a.sess.run(a.logits,feed_dict)


a.eval_grad()
a.v_grad_max


a.sess.run(a.softmax,feed_dict)
a.sess.run(a.loss,feed_dict)
