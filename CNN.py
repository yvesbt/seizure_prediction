from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import device_lib 


def CNN(x, cfg):
  print(x)
  conv1 = tf.layers.conv2d(
      inputs=x,
      filters=5,
      strides=1,
      kernel_size=[1,13],
      padding="valid",
      data_format='channels_first',
      activation=tf.nn.relu)
  print(conv1)
  pool1 = tf.layers.average_pooling2d(
      conv1,
      pool_size=[1,2],
      strides=[1,2],
      padding="valid",
      data_format='channels_first')
  print(pool1)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=5,
      strides=1,
      kernel_size=[cfg.num_input, 9],
      padding="valid",
      data_format='channels_first',
      activation=tf.nn.relu)
  print(conv2)
  pool2 = tf.layers.average_pooling2d(
      conv2,
      pool_size=[1,2],
      strides=[1,2],
      padding="valid",
      data_format='channels_first')
  print(pool2)
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=3,
      strides=1,
      kernel_size=[1, 8],
      padding="valid",
      activation=tf.nn.relu,
      data_format='channels_first')
  conv3=tf.reshape(conv3,[1,1,3])
  output = tf.layers.dense(conv3, cfg.num_classes, activation=None)
  return output
