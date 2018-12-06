from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib 
import pyedflib
import cProfile
import glob
import re
import numpy as np
import random
import EEG
import config
import FC
import TCN
import RNN

flags = tf.flags
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("NN", None,
                    "Type of neural network.")
flags.DEFINE_string("patient", None,
                    "Patient number")
FLAGS = flags.FLAGS                   
cfg = config.Config(data_path = FLAGS.data_path, NN = FLAGS.NN, patient = int(FLAGS.patient))
patient_data = EEG.Patient_data(cfg)

# tf Graph input
X = tf.placeholder("float", [None, cfg.timesteps, cfg.num_input])
Y = tf.placeholder("float", [None, cfg.num_classes])

if (cfg.NN == "TCN"):
  logits = TCN.TCN(X, cfg)
elif (cfg.NN == "FC"):
  logits = FC.FC(X, cfg)
elif (cfg.NN == "RNN"):
  logits = RNN.RNN(X, cfg)

# ~ prediction = tf.nn.sigmoid(logits)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y)) + tf.losses.get_regularization_loss()
# ~ loss_op = tf.losses.sigmoid_cross_entropy(multi_class_labels = Y, logits = logits)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(cfg.learning_rate).minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
save_file = "./ckpt/"+cfg.NN+".ckpt"
# Start trainingrun 
# ~ with tf.Session() as sess:
def run():
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16,inter_op_parallelism_threads=16)) as sess:
    
    # Run the initializer
    sess.run(init)

    for step in range(1, cfg.training_steps+1):
      batch_x, batch_y = patient_data.train_next_batch(cfg.batch_size, cfg.num_input)
      batch_x = batch_x.transpose((0,2,1))
      sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
      if step % cfg.display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                   Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
                    "{:.4f}".format(loss) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
    save_path = saver.save(sess, save_file)
    print("Model saved in path: %s" % save_path)
    print("Optimization Finished!")
    
    test_data, test_label = patient_data.get_test_batch(cfg.test_len, cfg.num_input)
    # ~ test_data, test_label = patient_data.train_next_batch(cfg.test_len, cfg.num_input)
    test_data = test_data.transpose((0,2,1))
    acc, pred = sess.run([accuracy, prediction], feed_dict={X: test_data, Y: test_label})
    # ~ print(pred)
    print("Testing Accuracy:" + "{:.3f}".format(acc))

run()
print(cfg.segments_type_train,cfg.segments_type_test)
# ~ cProfile.run('run()')
