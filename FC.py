from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib 

'''
  A simple Fully Connected Neural network with two layers, dropout and regularizers
'''
def FC(x, cfg):
  inputs_1D = tf.reshape(x,[-1,cfg.num_inputs*cfg.N_features])
  first_layer = tf.layers.dense(inputs_1D, cfg.num_hidden, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
  first_layer = tf.layers.dropout(first_layer, rate=0.2)
  second_layer = tf.layers.dense(first_layer, cfg.num_classes, activation=None, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
  return second_layer
