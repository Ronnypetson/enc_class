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
patch2_upper_left = (0,0)
patch2_bottom_right = (data_dim[0]-1,data_dim[1]-1)

# Get dimensions of each patch
patch1_dim = [patch1_bottom_right[0]-patch1_upper_left[0]+1, patch1_bottom_right[1]-patch1_upper_left[1]+1]
patch2_dim = [patch2_bottom_right[0]-patch2_upper_left[0]+1, patch2_bottom_right[1]-patch2_upper_left[1]+1]

batch_size = 50
learning_rate = 0.001
model_fn = "checkpoint/auto_enc/auto_enc.ckpt"

def norm_batch(batch):
  batch = batch.reshape((batch.shape[0],data_dim[0],data_dim[1],1))
  x = np.zeros([batch.shape[0],patch1_dim[0],patch1_dim[1],1])
  y = np.zeros([batch.shape[0],patch2_dim[0],patch2_dim[1],1])
  for i in range(batch.shape[0]):
    ul = patch1_upper_left
    br = patch1_bottom_right
    x[i] = batch[i][ul[0]:br[0]+1,ul[1]:br[1]+1]
    ul = patch2_upper_left
    br = patch2_bottom_right
    y[i] = batch[i][ul[0]:br[0]+1,ul[1]:br[1]+1]
  return x,y

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 2x upsample via transposed convolution ('deconvolution')
def deconv2d2x(x, W, output_shape):
  return tf.nn.conv2d_transpose(
          x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME')

# Encoder
X = tf.placeholder(tf.float32,[None,patch1_dim[0],patch1_dim[1],1],name='X')
with tf.variable_scope('conv1'):
  conv1 = tf.layers.conv2d(X,32,(3,3),padding='same',activation=tf.nn.relu)
with tf.variable_scope('conv2'):
  conv2 = tf.layers.conv2d(conv1,64,(3,3),padding='same',activation=tf.nn.relu)
with tf.variable_scope('fc1'): # Rep IN->OUT
  fc1_ = tf.contrib.layers.flatten(conv2)
  fc1 = tf.layers.dense(fc1_,100,activation=tf.nn.relu,name='fc1')
with tf.variable_scope('fc2'):
  fc2 = tf.layers.dense(fc1,patch1_dim[0]*patch1_dim[1]*64,activation=tf.nn.relu)
  fc2 = tf.reshape(fc2,[-1,patch1_dim[0],patch1_dim[1],64]) #
# Decoder
with tf.variable_scope('deconv1'):
  W_conv1 = weight_variable([5, 5, 32, 64])
  b_conv1 = bias_variable([32])
  oshape1 = [batch_size, patch2_dim[0], patch2_dim[1], 32]
  h_conv1 = deconv2d2x(fc2, W_conv1, oshape1) + b_conv1
with tf.variable_scope('deconv2'):
  W_conv2 = weight_variable([5, 5, 1, 32])
  b_conv2 = bias_variable([1])
  oshape2 = [batch_size, patch2_dim[0], patch2_dim[1], 1]
  h_conv2 = deconv2d2x(h_conv1, W_conv2, oshape2) + b_conv2

y_ = h_conv2 #tf.reshape(h_conv2,[-1,patch2_dim[0],patch2_dim[1],1])
Y = tf.placeholder(tf.float32,[None,patch2_dim[0],patch2_dim[1],1])
cost = tf.losses.mean_squared_error(Y,y_)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
  saver = tf.train.Saver()
  if os.path.isfile(model_fn+'.meta'):
    saver.restore(sess,model_fn)
  else:
    sess.run(tf.global_variables_initializer())
  for i in range(200):
    batch = mnist.train.next_batch(batch_size)
    b_x, b_y = norm_batch(batch[0])
    batch_cost, _ = sess.run([cost, train], feed_dict={X: b_x,Y: b_y})
    print ('batch %s cost: %s' % (i, batch_cost))
  saver.save(sess,model_fn)
  batch = mnist.test.next_batch(batch_size)
  b_x, b_y = norm_batch(batch[0])
  y_img = sess.run(y_, feed_dict={X:b_x})
  y_img = y_img.reshape((batch_size, patch2_dim[0], patch2_dim[1]))
  for i in range(3):
    plt.figure()
    plt.imshow(b_y[i].reshape(patch2_dim),cmap='gray')
    plt.figure()
    plt.imshow(y_img[i],cmap='gray')
  plt.show()

