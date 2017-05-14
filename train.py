import tensorflow as tf
import numpy as np
import time
import math

from dataloader import DataLoader

# configuration
BATCH_SIZE = 50
TRAIN_SET_PATH = "./data/train_32x32.mat"
MAX_EPOCHS = 100
LEARNING_RATE = 0.01

# define model
''' 
  input:    [batch, in_height, in_width, in_channels]
  filter:   [filter_height, filter_width, in_channels, out_channels]
  strides:  [1, stride_h, stride_w, 1]
  padding:  "VALID" - no padding
            "SAME" - zero padding
'''
def conv2d(input_data, num_out_channels, width, height, strides, padding, scope):
  with tf.variable_scope(scope):
    weights = tf.get_variable("w", [height, width, input_data.get_shape()[-1], num_out_channels])
    bias = tf.get_variable("b", [num_out_channels])
    # TODO: activation function?
  return tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=strides, padding=padding), bias)

# TODO: computation
def fc(input_data, num_neurons, scope):
  shape = intput_data.get_shape().as_list()
  dim = 1
  for d in shape[1:]:
    dim *= d
  input_data = tf.reshape(input_data, [-1, dim])
  with tf.variable_scope(scope):
    weights = tf.get_variable("w", [input_data.get_shape()[-1]])
    bias = tf.get_variable("b", [num_neurons])
  # TODO: activation function?
  return tf.nn.bias_add(tf.matmul(input_data, weights), bias)

def loss(input_data, scope):
  with tf.variable_scope(scope):
    
  return 

def update(loss, ):
  with tf.variable_scope(scope):
    learning_rate = tf.get_variable(learning_rate, trainable=False, name='learning_rate')
    tvars = tf.trainable_variables()
    '''
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    '''
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
  return 

def SVHN():
  # convolution section 1
  conv = conv2d()
  pool = tf.nn.max_pool()
  # fully connected section 1
  logits = fc()
  # loss section
  loss = loss(logits)
  # training section
  update(loss, learning_rate)  
  

# train model
def main(_):
  train_set_loader = DataLoader(TRAIN_SET_PATH, BATCH_SIZE)
  valid_set_loader = DataLoader(VALID_SET_PATH)
  
  with tf.Graph().as_default(), tf.Session() as session:
    tf.set_random_seed(time.time())
    initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=)
    with tf.variable_scope("SVHN", initializer=initializer):
      train_model = SVHN()
    saver = tf.train.Saver()
    with tf.variable_scope("SVHN", reuse=True):
      valid_model = SVHN()
    
    tf.global_vairables_initializer().run()
    
    for epoch in range(MAX_EPOCHS):
      for image_batch, label_batch in train_set_loader.iter():
        session.run([], {})
        

if __name__ == "__main__":
  tf.app.run()
