import tensorflow as tf
import numpy as np
import math

# define model
class SVHN:
  def __init__(self):

  '''
    perform prediction
  '''
  # TODO: nomailize result using softmax
  def inference(self):
    inputs = tf.placeholder(tf.int64, shape=[batch_size, 32, 32], name="input")
    # conv1: convolution and rectified linear activation.
    conv1 = self.conv2d(inputs, 5, 5, 64, scope="conv1")
    # pool1: max pooling.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    # norm1: local response normalization.
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    # conv2: convolution and rectified linear activation.
    conv2 = self.conv2d(norm1, 5, 5, 64, scope="conv2")
    # norm2: local response normalization.
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    # pool2: max pooling.
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    # fc1: fully connected layer with rectified linear activation.
    fc1 = self.fc(pool2, 384, scope="fc1")
    # fc2: fully connected layer with rectified linear activation.
    fc2 = self.fc(fc1, 192, scope="fc2")
    # softmax: linear transformation to produce logits.
    # softmax is NOT performed here for efficiency
    softmax_fc = self.fc(fc2, 10, scope="softmax_fc")
    return softmax_fc
    
  ''' 
    input:    [batch, in_height, in_width, in_channels]
    filter:   [filter_height, filter_width, in_channels, out_channels]
    padding:  "VALID" - no padding
              "SAME" - zero padding
  '''
  # TODO: add weight decay
  # TODO: specify initializer for weights and biases
  def conv2d(input_data, height, width, num_out_channels, stride=1, padding="SAME", activation=tf.nn.relu, weight_decay=True, scope="conv2d"):
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
      # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    return 

