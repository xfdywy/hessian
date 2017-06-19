from keras.datasets import cifar10
import numpy as np

import theano
import theano.tensor as T
from theano import function

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
x_train = np.transpose(X_train,(0,3,1,2))
x_test = np.transpose(X_train,(0,3,1,2))



