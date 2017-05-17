import tensorflow as tf
import numpy as np
import math

# define model
def inference(inputs):
  '''
    perform inference
  '''
  inputs = tf.stack(inputs, name="input_images_tensor")
  # TODO: nomailize result using softmax
  # conv1: convolution and rectified linear activation.
  conv1 = conv2d(inputs, 5, 5, 64, scope="conv1")
  # pool1: max pooling.
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
  # norm1: local response normalization.
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
  # conv2: convolution and rectified linear activation.
  conv2 = conv2d(norm1, 5, 5, 64, scope="conv2")
  # norm2: local response normalization.
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2: max pooling.
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
  # fc1: fully connected layer with rectified linear activation.
  fc1 = fc(pool2, 384, scope="fc1")
  # fc2: fully connected layer with rectified linear activation.
  fc2 = fc(fc1, 192, scope="fc2")
  # softmax: linear transformation to produce logits.
  # softmax is NOT performed here for efficiency
  logits = fc(fc2, 10, scope="softmax_fc")
  return logits
  
def conv2d(input_data, height, width, num_out_channels, stride=1, padding="SAME", activation=tf.nn.relu, weight_decay=True, scope="conv2d"):
  ''' 
    input:    [batch, in_height, in_width, in_channels]
    filter:   [filter_height, filter_width, in_channels, out_channels]
    padding:  "VALID" - no padding
              "SAME" - zero padding
  '''
  # TODO: add weight decay
  # TODO: specify initializer for weights and biases
  with tf.variable_scope(scope):
    weights = tf.get_variable("w", [height, width, input_data.get_shape()[-1], num_out_channels])
    bias = tf.get_variable("b", [num_out_channels])
    if activation:
      return activation(tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding=padding), bias))
    else:
      return tf.nn.bias_add(tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding=padding), bias)

def fc(input_data, output_dim, activation=tf.nn.relu, scope="fc"):
  ''' 
    input:    [batch, in_channels]
  '''
  shape = input_data.get_shape().as_list()
  with tf.variable_scope(scope):
    if len(shape) == 2:
      weights = tf.get_variable("w", [shape[1], output_dim])
      input_data = tf.reshape(input_data, [-1, weights.get_shape().as_list()[0]])
    elif len(shape) == 4:
      input_data = tf.reshape(input_data, [shape[0], -1])
      weights = tf.get_variable("w", [shape[1]*shape[2]*shape[3], output_dim])
    else:
      raise ValueError("Linear expects 2D/4D shape: %d" % len(shape))
    bias = tf.get_variable("b", [output_dim])
  if activation:
    return activation(tf.nn.bias_add(tf.matmul(input_data, weights), bias))
  else:
    return tf.nn.bias_add(tf.matmul(input_data, weights), bias)

def loss(logits, labels, scope="loss"):
  ''' 
    input:    [batch, classes]
  '''
  labels = tf.stack(labels, name="input_labels_tensor")
  with tf.variable_scope(scope):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
    loss = cross_entropy_mean
    # TODO: add weight decay loss
    # tf.add_to_collection("losses", cross_entropy_mean)
    # loss = tf.add_n(tf.get_collection("losses", name="total_loss")
  return loss

def optimize(loss, global_step, learning_rate=1.0, max_grad_norm=5.0, scope="optimize"):
  # for continuing training
  with tf.variable_scope(scope):
    tvars = tf.trainable_variables()
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    # TODO: other optimizers?
    # learning_rate = tf.get_variable(learning_rate, trainable=False, name='learning_rate')
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
  return train_op

