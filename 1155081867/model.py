from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

import dataloader

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = dataloader.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = dataloader.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# define model
def inference(inputs):
  '''
    perform inference
  '''
  inputs = tf.stack(inputs, name="input_images_tensor")
  # conv1: convolution and rectified linear activation.
  conv1 = conv2d(inputs, 5, 5, 64, 0.0, scope="conv1")
  # pool1: max pooling.
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
  # norm1: local response normalization.
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
  # conv2: convolution and rectified linear activation.
  conv2 = conv2d(norm1, 5, 5, 64, 0.1, scope="conv2")
  # norm2: local response normalization.
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2: max pooling.
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
  # fc1: fully connected layer with rectified linear activation.
  fc1 = fc(pool2, 384, 0.1, scope="fc1")
  # fc2: fully connected layer with rectified linear activation.
  fc2 = fc(fc1, 192, 0.1, scope="fc2")
  # softmax: linear transformation to produce logits.
  # softmax is NOT performed here for efficiency
  logits = fc(fc2, 10, 0.0, stddev=1/192.0, wd=0.0, activation=None, scope="softmax_fc")
  return logits
  
def conv2d(input_data, height, width, num_out_channels, bias_init, stride=1, padding="SAME", activation=tf.nn.relu, weight_decay=True, scope="conv2d"):
  ''' 
    input:    [batch, in_height, in_width, in_channels]
    filter:   [filter_height, filter_width, in_channels, out_channels]
    padding:  "VALID" - no padding
              "SAME" - zero padding
  '''
  with tf.variable_scope(scope):
    weights = get_decay_weight("w", shape=[height, width, input_data.get_shape()[-1], num_out_channels], stddev=5e-2, wd=0.0)
    # weights = tf.get_variable("w", [height, width, input_data.get_shape()[-1], num_out_channels])
    bias = tf.get_variable("b", [num_out_channels], initializer=tf.constant_initializer(bias_init), dtype=tf.float32)
    if activation:
      return activation(tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding=padding), bias))
    else:
      return tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding=padding), bias)

def fc(input_data, output_dim, bias_init, stddev=0.04, wd=0.004, activation=tf.nn.relu, scope="fc"):
  ''' 
    input:    [batch, in_channels]
  '''
  shape = input_data.get_shape().as_list()
  with tf.variable_scope(scope):
    if len(shape) == 2:
      # input_data = tf.reshape(input_data, [-1, weights.get_shape().as_list()[0]])
      weights = get_decay_weight("w", shape=[shape[1], output_dim], stddev=stddev, wd=wd)
      # weights = tf.get_variable("w", [shape[1], output_dim])
    elif len(shape) == 4:
      input_data = tf.reshape(input_data, [shape[0], -1])
      weights = get_decay_weight("w", shape=[input_data.get_shape()[1].value, output_dim], stddev=stddev, wd=wd)
      # weights = tf.get_variable("w", [shape[1]*shape[2]*shape[3], output_dim])
    else:
      raise ValueError("Linear expects 2D/4D shape: %d" % len(shape))
    bias = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(bias_init), dtype=tf.float32)
  if activation:
    return activation(tf.matmul(input_data, weights) + bias, name=scope)
  else:
    return tf.add(tf.matmul(input_data, weights), bias, name=scope)

def loss(logits, labels, scope="loss"):
  ''' 
    input:    [batch, classes]
  '''
  labels = tf.stack(labels, name="input_labels_tensor")
  labels = tf.cast(labels, tf.int64)
  with tf.variable_scope(scope):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
    # loss = cross_entropy_mean
    tf.add_to_collection("losses", cross_entropy_mean)
    loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
  return loss

def optimize(loss, global_step, init_learning_rate=0.1, batch_size=128, max_grad_norm=5.0, scope="optimize"):
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # for continuing training
  with tf.variable_scope(scope):
    # learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
    learning_rate = tf.train.exponential_decay(init_learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # TODO: other optimizers?
    # tvars = tf.trainable_variables()
    # grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    loss_averages_op = add_loss_summaries(loss)
    with tf.control_dependencies([loss_averages_op]):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
  return train_op

def evaluate(session, top_k_op, num_examples):
  predictions = session.run([top_k_op])
  true_count = np.sum(predictions)
  precision = float(true_count) / num_examples
  if precision >= .9:
    print(predictions)
  return precision * 100

def get_var_cpu(name, shape, initializer):
  # with tf.device('/cpu:0'):
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var

def get_decay_weight(name, shape, stddev, wd):
  var = get_var_cpu(name, shape,
    tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def add_loss_summaries(loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [loss])
  return loss_averages_op
