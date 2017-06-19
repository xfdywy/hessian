# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:54:34 2017

@author: v-yuewng
"""

x = np.random.random([4,32,32,3])
y = np.array([1,2,3,4])

model = cifarnet()
sess = tf.Session()
model.buildnet()
sess.run(model.init_allvars)

sess.run(model.meanloss,feed_dict={model.images : x, model.label : y, model.learningrate : 100 , model.dropout_keep_prob : 1})    


for ii in range(1000):
    sess.run([model.train,model.meanloss],feed_dict={model.images : x, model.label : y, model.learningrate : 0.0001 , model.dropout_keep_prob : 1})
    
sess.run(model.meanloss,feed_dict={model.images : x, model.label : y, model.learningrate : 0.5 , model.dropout_keep_prob : 1})    
    
dloss_dw = tf.gradients(model.meanloss,model.allvars)[0]
hess =[]
ind = np.random.randint(0,173578,[100])
for i in range(100):
    
    dfx_i = tf.slice(dloss_dw,[i],[1])
    ddfx_i = tf.gradients(dfx_i,model.allvars)[0]
    hess.append(ddfx_i)
    
hess = tf.squeeze(hess) 

v_grad = sess.run(dloss_dw,feed_dict={model.images : x, model.label : y, model.learningrate : 0.5 , model.dropout_keep_prob : 1})

v_hess = sess.run(hess,feed_dict={model.images : x, model.label : y, model.learningrate : 0.5 , model.dropout_keep_prob : 1})


np.max(v_hess)
np.min(v_hess)


np.max(v_grad)
np.min(v_grad)
np.sum(v_grad**2)


plt.hist(v_hess)
plt.hist(v_grad)



from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

data_num = X_train.shape[0]
minibatchsize = 64
for ii in range(1000):

    sample = np.random.randint(0,data_num,[minibatchsize])
    x = X_train[sample,:,:,:]
    y = y_train[sample,0]
    
    sess.run([model.train,model.meanloss],feed_dict={model.images : x, model.label : y, model.learningrate : 0.000001 , model.dropout_keep_prob : 1})
    






v_grad = sess.run(dloss_dw,feed_dict={model.images : x, model.label : y, model.learningrate : 0.5 , model.dropout_keep_prob : 1})

v_hess = sess.run(hess,feed_dict={model.images : x, model.label : y, model.learningrate : 0.5 , model.dropout_keep_prob : 1})


np.max(v_hess)
np.min(v_hess)


np.max(v_grad)
np.min(v_grad)
np.sum(v_grad**2)


plt.hist(np.ravel(v_hess),500);plt.show()

plt.hist(v_grad,100);plt.show()




