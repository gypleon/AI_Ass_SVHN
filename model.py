import tensorflow as tf
import numpy as np
import math

# define model
''' 
  input:    [batch, in_height, in_width, in_channels]
  filter:   [filter_height, filter_width, in_channels, out_channels]
  padding:  "VALID" - no padding
            "SAME" - zero padding
'''
def conv2d(input_data, num_out_channels, width, height, stride=1, padding="SAME", activation=tf.nn.relu, scope="conv2d"):
  with tf.variable_scope(scope):
    weights = tf.get_variable("w", [height, width, input_data.get_shape()[-1], num_out_channels])
    bias = tf.get_variable("b", [num_out_channels])
    if activation:
      return activation(tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding=padding), bias))
    else:
      return tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding=padding), bias)

''' 
  input:    [batch, in_channels]
'''
def fc(input_data, num_neurons, activation=tf.nn.relu, scope="fc"):
  shape = input_data.get_shape().as_list()
  with tf.variable_scope(scope):
    if len(shape) == 2:
      weights = tf.get_variable("w", [shape[1], num_neurons])
      input_data = tf.reshape(input_data, [-1, weights.get_shape().as_list()[0])
    elif len(shape) == 4:
      weights = tf.get_variable("w", [shape[1]*shape[2]*shape[3], num_neurons])
    else:
      raise ValueError("Linear expects 2D/4D shape: %d" % len(shape))
    bias = tf.get_variable("b", [num_neurons])
  if activation:
    return activation(tf.nn.bias_add(tf.matmul(input_data, weights), bias))
  else:
    return tf.nn.bias_add(tf.matmul(input_data, weights), bias)

''' 
  input:    [batch, in_channels]
'''
def loss(logits, batch_size, scope="loss"):
  with tf.variable_scope(scope):
    labels = tf.placeholder(tf.int64, shape=[batch_size], name="labels")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")
  return loss

def optimize(loss, learning_rate=1.0, scope="optimize"):
  # for continuing training
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope(scope):
    learning_rate = tf.get_variable(learning_rate, trainable=False, name='learning_rate')
    tvars = tf.trainable_variables()
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    # TODO: other optimizers?
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
  return 

def SVHN(inputs, batch_size, scope="SVHN"):
  inputs = tf.placeholder(tf.int64, shape=[batch_size, 32, 32], name="input")
  # convolution section 1
  conv1 = conv2d()
  pool = tf.nn.max_pool()
  # fully connected section 1
  logits = fc()
  # loss section
  loss = loss(logits)
  
class SVHN:
  def __init__(self):

  def inference(self):

  def optimize(self):
