import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Data dimensions
data_dim = (28,28)

# Point coordinates of the input and output patches
patch1_upper_left = (0,0)
patch1_bottom_right = (data_dim[0]-1,data_dim[1]-1)

# Get dimensions of each patch
patch1_dim = [patch1_bottom_right[0]-patch1_upper_left[0]+1, patch1_bottom_right[1]-patch1_upper_left[1]+1]

batch_size = 50
learning_rate = 0.001
num_epochs = 200
model_fn = "checkpoint/enc_class/enc_class.ckpt"
prev_model = "checkpoint/auto_enc/auto_enc.ckpt"

def get_rep(sess,b_x):
    return sess.run('fc1/fc1/Relu:0',feed_dict={'X:0':b_x})

tf.reset_default_graph()
with tf.Session() as sess:
  if os.path.isfile(prev_model+'.meta'):
    saver = tf.train.import_meta_graph(prev_model+'.meta')
    saver.restore(sess,prev_model)
  else:
    print("No model to load")
  batches = []
  for i in range(num_epochs):
    b_x, b_y = mnist.train.next_batch(batch_size)
    b_x = get_rep(sess,b_x.reshape(batch_size,data_dim[0],data_dim[1],1))
    batches.append([b_x,b_y])

# Classifier from encoding
tf.reset_default_graph()
X = tf.placeholder(tf.float32,[None,100]) # middle representation from autoencoder
with tf.variable_scope('fc1_'):
  fc1 = tf.layers.dense(X,40,activation=tf.nn.relu)
with tf.variable_scope('fc2_'):
  y_ = tf.layers.dense(fc1,10,activation=tf.nn.relu)

Y = tf.placeholder(tf.float32,[None,10])
cost = tf.losses.mean_squared_error(Y,y_)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
  saver = tf.train.Saver()
  if os.path.isfile(model_fn+'.meta'):
    saver.restore(sess,model_fn)
  else:
    sess.run(tf.global_variables_initializer())
  for i in range(num_epochs):
    batch_cost, _ = sess.run([cost, train], feed_dict={X: batches[i][0],Y: batches[i][1]})
    print ('batch %s cost: %s' % (i, batch_cost))
  saver.save(sess,model_fn)

