from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib 


def RNN(x, cfg):
  # GPU version
  # ~ lstm_cuda = tf.contrib.cudnn_rnn.CudnnLSTM(1,num_hidden)
  # ~ outputs, _ = lstm_cuda(x)
  # ~ lstm_cell = tf.contrib.rnn.LSTMBlockCell(num_hidden, forget_bias=1.0)
  # ~ lstm_cell = tf.contrib.rnn.LSTMCell(cfg.num_hidden, forget_bias=1.0)
  inputs_rs = tf.transpose(x, [0, 2, 1])
  stacked_rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(cfg.num_hidden, forget_bias=1.0) for _ in range(2)])
  outputs, _ = tf.nn.dynamic_rnn( cell=stacked_rnn, inputs=inputs_rs, dtype=tf.float32,  parallel_iterations=10)
  # we take only the output at the last time
  # ~ fc_layer = tf.layers.dense(outputs[:,-1,:], 30, activation=tf.nn.relu)
  # we take all outputs
  # ~ outputs_1D = tf.reshape(outputs,[-1,cfg.num_hidden*cfg.timesteps])
  outputs_1D = tf.reshape(outputs,[-1,cfg.num_hidden*cfg.num_input])
  fc_layer = tf.layers.dense(outputs_1D, 30, activation=tf.nn.relu)
  output_layer = tf.layers.dense(fc_layer, cfg.num_classes, activation=None)
  return output_layer
